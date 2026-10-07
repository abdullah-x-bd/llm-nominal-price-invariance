from __future__ import annotations

from dataclasses import dataclass
from .scenarios import Scenario


@dataclass(frozen=True)
class Condition:
    condition_id: str
    ratio: int
    cheap_asset: str | None
    replicate: int | None = None


CONDITIONS = (
    Condition("neutral_r1", 1, None, 1),
    Condition("neutral_r2", 1, None, 2),
    Condition("2x_Acheap", 2, "A"),
    Condition("2x_Bcheap", 2, "B"),
    Condition("10x_Acheap", 10, "A"),
    Condition("10x_Bcheap", 10, "B"),
    Condition("50x_Acheap", 50, "A"),
    Condition("50x_Bcheap", 50, "B"),
)


def prices_for(s: Scenario, c: Condition) -> tuple[float, float]:
    if c.ratio == 1:
        return s.neutral_price, s.neutral_price
    low = {2: s.low_price_2x, 10: s.low_price_10x, 50: s.low_price_50x}[c.ratio]
    high = low * c.ratio
    if c.cheap_asset == "A":
        return low, high
    if c.cheap_asset == "B":
        return high, low
    raise ValueError(f"Invalid condition {c}")


def _asset_block(label: str, mu: float, sigma: float, price: float) -> str:
    return (
        f"Stock {label}: expected annual return {mu*100:.2f}%; "
        f"annual volatility {sigma*100:.2f}%; share price ${price:,.2f}."
    )


def build_prompt(s: Scenario, c: Condition) -> str:
    price_a, price_b = prices_for(s, c)
    blocks = {
        "A": _asset_block("A", s.mu_a, s.sigma_a, price_a),
        "B": _asset_block("B", s.mu_b, s.sigma_b, price_b),
    }
    ordered = [blocks[ch] for ch in s.display_order]
    return "\n".join([
        "Allocate a $10,000 long-only portfolio between two fictional listed equities.",
        "Fractional shares are available. There are no fees, taxes, lot-size constraints, short sales, leverage, or liquidity differences. Use only the information supplied.",
        f"Objective: maximize mean-variance utility U = E[R_p] - ({s.gamma:.1f}/2) Var(R_p). Use decimal returns in the utility formula, so 10% means 0.10.",
        ordered[0],
        ordered[1],
        f"Return correlation: {s.rho:.2f}.",
        'Return only JSON with integer percentage weights that sum to 100, exactly like {"A":50,"B":50}.',
    ])
