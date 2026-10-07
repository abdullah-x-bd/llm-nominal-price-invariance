from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

from .config import CONCURRENCY, LIVE_RUN_STOP_USD, MODELS
from .openrouter import OpenRouterClient
from .prompts import CONDITIONS, build_prompt, prices_for
from .scenarios import Scenario, generate_scenarios


def _scenario_from_row(row: dict[str, str]) -> Scenario:
    numeric = {
        "mu_a", "mu_b", "sigma_a", "sigma_b", "rho", "gamma",
        "optimal_weight_a", "neutral_price", "low_price_2x", "low_price_10x", "low_price_50x"
    }
    kwargs: dict[str, Any] = {}
    for k, v in row.items():
        kwargs[k] = float(v) if k in numeric else v
    return Scenario(**kwargs)


def load_scenarios(path: Path | None) -> list[Scenario]:
    if path is None:
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
        scenarios = [next(s for s in scenarios if s.scenario_type == "symmetric"),
                     next(s for s in scenarios if s.scenario_type == "asymmetric")]
    elif mode != "full":
        raise ValueError("mode must be pilot or full")

    jobs = [(s, c, m) for s in scenarios for c in CONDITIONS for m in MODELS]
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
    stop = False
    completed = 0
    start = time.time()

    async def do_job(s, c, m):
        nonlocal estimated_spend, stop, completed
        async with sem:
            async with lock:
                if stop:
                    return
                worst_next = 450 / 1_000_000 * m.input_per_million + 32 / 1_000_000 * m.output_per_million
                if estimated_spend + worst_next >= LIVE_RUN_STOP_USD:
                    stop = True
                    return
            prompt = build_prompt(s, c)
            pa, pb = prices_for(s, c)
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
                    **result,
                    "cost_usd": float(cost or 0.0),
                }
                async with lock:
                    rows.append(row)
                    estimated_spend += float(cost or 0.0)
                    completed += 1
                    if completed % 100 == 0:
                        info = await client.key_info()
                        actual = float(info.get("usage") or 0.0) - baseline_usage
                        if actual >= LIVE_RUN_STOP_USD:
                            stop = True
            except Exception as exc:
                async with lock:
                    failures.append({
                        "scenario_id": s.scenario_id,
                        "condition_id": c.condition_id,
                        "model": m.slug,
                        "error": repr(exc),
                    })

    try:
        await asyncio.gather(*(do_job(*job) for job in jobs))
        ending = await client.key_info()
    finally:
        await client.close()

    actual_spend = float(ending.get("usage") or 0.0) - baseline_usage
    if actual_spend > 1.35 + 1e-6:
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
        "n_planned": len(jobs),
        "n_completed": len(rows),
        "n_failures": len(failures),
        "estimated_spend_usd": round(estimated_spend, 6),
        "actual_key_spend_usd": round(actual_spend, 6),
        "baseline_key_usage_usd": baseline_usage,
        "elapsed_seconds": round(time.time() - start, 2),
        "stopped_by_budget_guard": stop,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
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
