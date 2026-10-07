from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from .config import CONCURRENCY, HARD_USER_BUDGET_USD, LIVE_RUN_STOP_USD, MODELS
from .openrouter import OpenRouterClient
from .prompts import CONDITIONS, build_prompt, prices_for
from .scenarios import Scenario, generate_scenarios

REQUEST_ORDER_SEED = 20261008


def _scenario_from_row(row: dict[str, str]) -> Scenario:
    numeric = {
        "mu_a", "mu_b", "sigma_a", "sigma_b", "rho", "gamma",
        "optimal_weight_a", "neutral_price", "low_price_2x", "low_price_10x", "low_price_50x",
    }
    kwargs: dict[str, Any] = {}
    for k, v in row.items():
        kwargs[k] = float(v) if k in numeric else v
    return Scenario(**kwargs)


def load_scenarios(path: Path | None) -> list[Scenario]:
    if path is None or not path.exists():
        return generate_scenarios()
    with path.open(encoding="utf-8") as f:
        return [_scenario_from_row(r) for r in csv.DictReader(f)]


def estimated_cost(spec, prompt_tokens: int | None, completion_tokens: int | None) -> float:
    pin = prompt_tokens or 0
    pout = completion_tokens or 0
    return pin / 1_000_000 * spec.input_per_million + pout / 1_000_000 * spec.output_per_million


async def run(mode: str, out: Path, scenarios_path: Path | None, concurrency: int) -> None:
    scenarios = load_scenarios(scenarios_path)
    if mode == "pilot":
        scenarios = [
            next(s for s in scenarios if s.scenario_type == "symmetric"),
            next(s for s in scenarios if s.scenario_type == "asymmetric"),
        ]
    elif mode != "full":
        raise ValueError("mode must be pilot or full")

    jobs = [(s, c, m) for s in scenarios for c in CONDITIONS for m in MODELS]
    random.Random(REQUEST_ORDER_SEED).shuffle(jobs)
    out.parent.mkdir(parents=True, exist_ok=True)
    failures_path = out.with_suffix(".failures.jsonl")

    client = OpenRouterClient()
    baseline = await client.key_info()
    baseline_usage = float(baseline.get("usage") or 0.0)
    if baseline.get("limit_remaining") is not None and float(baseline["limit_remaining"]) < 0.02:
        raise RuntimeError("OpenRouter key has insufficient remaining limit")

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    estimated_spend = 0.0
    reserved_spend = 0.0
    stop = False
    completed = 0
    authoritative_checks: list[dict[str, Any]] = []
    start = time.time()

    async def meter_check() -> None:
        nonlocal stop
        try:
            info = await client.key_info()
            actual = float(info.get("usage") or 0.0) - baseline_usage
            authoritative_checks.append({
                "completed": completed,
                "incremental_usage_usd": actual,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            if actual >= LIVE_RUN_STOP_USD:
                stop = True
        except Exception as exc:
            authoritative_checks.append({
                "completed": completed,
                "meter_error": repr(exc),
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            })

    async def do_job(s, c, m):
        nonlocal estimated_spend, reserved_spend, stop, completed
        worst_next = 450 / 1_000_000 * m.input_per_million + 32 / 1_000_000 * m.output_per_million
        async with sem:
            async with lock:
                if stop:
                    return
                if estimated_spend + reserved_spend + worst_next >= LIVE_RUN_STOP_USD:
                    stop = True
                    return
                reserved_spend += worst_next

            prompt = build_prompt(s, c)
            pa, pb = prices_for(s, c)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            try:
                result = await client.complete(m, prompt)
                cost = result.get("reported_cost")
                if cost is None:
                    cost = estimated_cost(m, result.get("prompt_tokens"), result.get("completion_tokens"))
                row = {
                    "scenario_id": s.scenario_id,
                    "scenario_type": s.scenario_type,
                    "condition_id": c.condition_id,
                    "ratio": c.ratio,
                    "cheap_asset": c.cheap_asset,
                    "model": m.slug,
                    "family": m.family,
                    "price_a": pa,
                    "price_b": pb,
                    "optimal_weight_a": s.optimal_weight_a,
                    "mu_a": s.mu_a,
                    "mu_b": s.mu_b,
                    "sigma_a": s.sigma_a,
                    "sigma_b": s.sigma_b,
                    "rho": s.rho,
                    "gamma": s.gamma,
                    "display_order": s.display_order,
                    "prompt_sha256": prompt_hash,
                    "prompt_text": prompt,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    **result,
                    "cost_usd": float(cost or 0.0),
                }
                should_check = False
                async with lock:
                    rows.append(row)
                    estimated_spend += float(cost or 0.0)
                    reserved_spend = max(0.0, reserved_spend - worst_next)
                    completed += 1
                    should_check = completed % 100 == 0
                if should_check:
                    await meter_check()
            except Exception as exc:
                async with lock:
                    reserved_spend = max(0.0, reserved_spend - worst_next)
                    failures.append({
                        "scenario_id": s.scenario_id,
                        "condition_id": c.condition_id,
                        "model": m.slug,
                        "prompt_sha256": prompt_hash,
                        "prompt_text": prompt,
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": repr(exc),
                    })

    ending: dict[str, Any] | None = None
    try:
        await asyncio.gather(*(do_job(*job) for job in jobs))
        try:
            ending = await client.key_info()
        except Exception as exc:
            authoritative_checks.append({"final_meter_error": repr(exc)})
    finally:
        await client.close()

    actual_spend = None
    if ending is not None:
        actual_spend = float(ending.get("usage") or 0.0) - baseline_usage
        if actual_spend > HARD_USER_BUDGET_USD + 1e-6:
            raise RuntimeError(f"Hard budget breached: ${actual_spend:.4f}")

    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else []
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["scenario_id"], r["condition_id"], r["model"])))
    with failures_path.open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, sort_keys=True) + "\n")

    manifest = {
        "mode": mode,
        "request_order_seed": REQUEST_ORDER_SEED,
        "n_planned": len(jobs),
        "n_completed": len(rows),
        "n_failures": len(failures),
        "estimated_spend_usd": round(estimated_spend, 6),
        "actual_key_spend_usd": round(actual_spend, 6) if actual_spend is not None else None,
        "baseline_key_usage_usd": baseline_usage,
        "authoritative_meter_checks": authoritative_checks,
        "elapsed_seconds": round(time.time() - start, 2),
        "stopped_by_budget_guard": stop,
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    p.add_argument("--out", type=Path, default=Path("data/results/results.csv"))
    p.add_argument("--scenarios", type=Path, default=Path("data/design/scenarios.csv"))
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = p.parse_args()
    asyncio.run(run(args.mode, args.out, args.scenarios, args.concurrency))


if __name__ == "__main__":
    main()
