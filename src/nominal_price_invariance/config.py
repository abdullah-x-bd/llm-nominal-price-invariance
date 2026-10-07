from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    family: str
    input_per_million: float
    output_per_million: float


# Prices verified from OpenRouter on 2026-10-07. Runtime cost controls are
# conservative and do not depend on these estimates alone.
MODELS = (
    ModelSpec("openai/gpt-5.6-luna", "OpenAI", 0.20, 1.20),
    ModelSpec("deepseek/deepseek-v4-flash", "DeepSeek", 0.04998, 0.09996),
    ModelSpec("z-ai/glm-5.3-flash", "Z.ai", 0.075, 0.25),
    ModelSpec("qwen/qwen3.5-397b-a17b", "Qwen", 0.39, 2.34),
    ModelSpec("meta-llama/llama-4-maverick", "Meta", 0.1875, 0.6525),
)

HARD_USER_BUDGET_USD = 1.35
LIVE_RUN_STOP_USD = 1.20
MAX_TOKENS = 32
CONCURRENCY = 16
