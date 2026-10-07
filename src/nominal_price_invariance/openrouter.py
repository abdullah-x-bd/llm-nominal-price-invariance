from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import MAX_TOKENS, ModelSpec

BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ParsedAllocation:
    weight_a: int
    weight_b: int


def parse_allocation(text: str) -> ParsedAllocation:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{[^{}]+\}", text)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            a = int(round(float(obj["A"])))
            b = int(round(float(obj["B"])))
            if not (0 <= a <= 100 and 0 <= b <= 100 and a + b == 100):
                raise ValueError("weights must be in [0,100] and sum to 100")
            return ParsedAllocation(a, b)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse allocation from {text!r}: {last_error}")


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/abdullah-x-bd/llm-nominal-price-invariance",
                "X-OpenRouter-Title": "LLM Nominal Price Invariance",
                "X-OpenRouter-Metadata": "enabled",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def key_info(self) -> dict[str, Any]:
        r = await self.client.get(f"{BASE_URL}/key")
        r.raise_for_status()
        return r.json()["data"]

    async def complete(self, model: ModelSpec, prompt: str, *, max_retries: int = 3) -> dict[str, Any]:
        payload = {
            "model": model.slug,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "reasoning": {"effort": "none", "exclude": True},
            "usage": {"include": True},
            "provider": {"sort": "price", "allow_fallbacks": True},
        }
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                r = await self.client.post(f"{BASE_URL}/chat/completions", json=payload)
                if r.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                body = r.json()
                content = body["choices"][0]["message"]["content"]
                parsed = parse_allocation(content)
                usage = body.get("usage") or {}
                return {
                    "response_id": body.get("id"),
                    "served_model": body.get("model"),
                    "provider": body.get("provider") or (body.get("openrouter_metadata") or {}).get("provider"),
                    "service_tier": body.get("service_tier"),
                    "raw_text": content,
                    "weight_a": parsed.weight_a,
                    "weight_b": parsed.weight_b,
                    "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
                    "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
                    "reasoning_tokens": ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                                         or (usage.get("output_tokens_details") or {}).get("reasoning_tokens")),
                    "reported_cost": usage.get("cost"),
                    "cache_discount": usage.get("cache_discount"),
                }
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                await asyncio.sleep(min(8.0, 0.75 * (2 ** attempt)))
        raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")
