from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

SEED = 20261007
GAMMA = 3.0
N_SYMMETRIC = 100
N_ASYMMETRIC = 200


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    scenario_type: str
    mu_a: float
    mu_b: float
    sigma_a: float
    sigma_b: float
    rho: float
    gamma: float
    optimal_weight_a: float
    display_order: str
    neutral_price: float
    low_price_2x: float
    low_price_10x: float
    low_price_50x: float


def optimal_weight_a(mu_a: float, mu_b: float, sigma_a: float, sigma_b: float,
                     rho: float, gamma: float = GAMMA) -> float:
    """Closed-form long-only optimum for two-asset mean-variance utility.

    U = E[R_p] - gamma/2 * Var(R_p)
    """
    va = sigma_a ** 2
    vb = sigma_b ** 2
    cov = rho * sigma_a * sigma_b
    denom = va + vb - 2 * cov
    if denom <= 0:
        raise ValueError("Non-positive variance denominator")
    w = ((mu_a - mu_b) / gamma - cov + vb) / denom
    return min(1.0, max(0.0, w))


def _round4(x: float) -> float:
    return round(x, 4)


def generate_scenarios(seed: int = SEED) -> list[Scenario]:
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    neutral_choices = [25.0, 50.0, 100.0, 200.0]
    low_choices = [5.0, 10.0, 20.0, 25.0, 40.0, 50.0, 80.0]

    # Symmetric states have an exact normative optimum of 50/50.
    for i in range(N_SYMMETRIC):
        mu = _round4(rng.uniform(0.06, 0.15))
        sigma = _round4(rng.uniform(0.12, 0.35))
        rho = _round4(rng.uniform(-0.20, 0.80))
        scenarios.append(Scenario(
            scenario_id=f"S{i+1:03d}",
            scenario_type="symmetric",
            mu_a=mu,
            mu_b=mu,
            sigma_a=sigma,
            sigma_b=sigma,
            rho=rho,
            gamma=GAMMA,
            optimal_weight_a=0.5,
            display_order=rng.choice(["AB", "BA"]),
            neutral_price=rng.choice(neutral_choices),
            low_price_2x=rng.choice(low_choices),
            low_price_10x=rng.choice(low_choices),
            low_price_50x=rng.choice(low_choices),
        ))

    # Asymmetric states are rejection-sampled to ensure an interior optimum.
    accepted = 0
    attempts = 0
    while accepted < N_ASYMMETRIC:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Could not generate enough interior asymmetric states")
        base_mu = rng.uniform(0.07, 0.13)
        delta = rng.uniform(-0.025, 0.025)
        mu_a = _round4(base_mu + delta / 2)
        mu_b = _round4(base_mu - delta / 2)
        sigma_a = _round4(rng.uniform(0.12, 0.35))
        sigma_b = _round4(rng.uniform(0.12, 0.35))
        rho = _round4(rng.uniform(-0.20, 0.80))
        w = optimal_weight_a(mu_a, mu_b, sigma_a, sigma_b, rho, GAMMA)
        if not (0.15 <= w <= 0.85):
            continue
        accepted += 1
        scenarios.append(Scenario(
            scenario_id=f"A{accepted:03d}",
            scenario_type="asymmetric",
            mu_a=mu_a,
            mu_b=mu_b,
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            rho=rho,
            gamma=GAMMA,
            optimal_weight_a=round(w, 8),
            display_order=rng.choice(["AB", "BA"]),
            neutral_price=rng.choice(neutral_choices),
            low_price_2x=rng.choice(low_choices),
            low_price_10x=rng.choice(low_choices),
            low_price_50x=rng.choice(low_choices),
        ))

    assert len(scenarios) == N_SYMMETRIC + N_ASYMMETRIC
    return scenarios


def scenario_rows(scenarios: Iterable[Scenario]) -> list[dict]:
    return [asdict(s) for s in scenarios]


def write_scenarios_csv(path: str | Path, scenarios: Iterable[Scenario]) -> None:
    rows = scenario_rows(scenarios)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def canonical_sha256(scenarios: Iterable[Scenario]) -> str:
    payload = json.dumps(scenario_rows(scenarios), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
