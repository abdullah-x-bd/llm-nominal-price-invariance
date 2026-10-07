from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PAIRS = [
    (2, "2x_Acheap", "2x_Bcheap"),
    (10, "10x_Acheap", "10x_Bcheap"),
    (50, "50x_Acheap", "50x_Bcheap"),
]
EQUIVALENCE_MARGIN_PP = 2.0
BOOTSTRAP_DRAWS = 5000
RANDOMIZATION_DRAWS = 20000
ANALYSIS_SEED = 20261009


def utility(w: pd.Series | np.ndarray, mu_a, mu_b, sigma_a, sigma_b, rho, gamma):
    w = np.asarray(w, dtype=float)
    mu = w * np.asarray(mu_a) + (1 - w) * np.asarray(mu_b)
    cov = np.asarray(rho) * np.asarray(sigma_a) * np.asarray(sigma_b)
    var = (
        w**2 * np.asarray(sigma_a) ** 2
        + (1 - w) ** 2 * np.asarray(sigma_b) ** 2
        + 2 * w * (1 - w) * cov
    )
    return mu - 0.5 * np.asarray(gamma) * var


def _clean(x) -> np.ndarray:
    return np.asarray(pd.Series(x).dropna(), dtype=float)


def mean_ci(x, *, draws: int = BOOTSTRAP_DRAWS, seed: int = ANALYSIS_SEED) -> tuple[float, float]:
    x = _clean(x)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_t_p(x) -> float:
    x = _clean(x)
    if len(x) < 2:
        return np.nan
    if np.allclose(x, x[0]) and np.isclose(x[0], 0):
        return 1.0
    return float(stats.ttest_1samp(x, 0.0).pvalue)


def wilcoxon_p(x) -> float:
    x = _clean(x)
    if len(x) == 0:
        return np.nan
    if np.allclose(x, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def tost_p(x, margin: float = EQUIVALENCE_MARGIN_PP) -> float:
    """Two one-sided tests for equivalence inside [-margin, +margin]."""
    x = _clean(x)
    n = len(x)
    if n < 2:
        return np.nan
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if np.isclose(sd, 0):
        return 0.0 if -margin < mean < margin else 1.0
    se = sd / np.sqrt(n)
    df = n - 1
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    return max(p_lower, p_upper)


def sign_flip_p(x, *, draws: int = RANDOMIZATION_DRAWS, seed: int = ANALYSIS_SEED) -> float:
    """Monte Carlo paired randomization p-value under exchangeability of treatment labels."""
    x = _clean(x)
    if len(x) == 0:
        return np.nan
    observed = abs(float(x.mean()))
    if np.allclose(x, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = draws
    chunk = 2000
    while remaining:
        k = min(chunk, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, len(x)))
        means = np.abs((signs * x).mean(axis=1))
        extreme += int(np.sum(means >= observed - 1e-12))
        remaining -= k
    return float((extreme + 1) / (draws + 1))


def _effect_summary(d: np.ndarray, *, seed_offset: int = 0) -> dict:
    d = _clean(d)
    abs_d = np.abs(d)
    d_lo, d_hi = mean_ci(d, seed=ANALYSIS_SEED + seed_offset)
    a_lo, a_hi = mean_ci(abs_d, seed=ANALYSIS_SEED + 1000 + seed_offset)
    return {
        "n": len(d),
        "mean_directional_effect_pp": float(d.mean()) if len(d) else np.nan,
        "directional_ci95_lo": d_lo,
        "directional_ci95_hi": d_hi,
        "mean_absolute_shift_pp": float(abs_d.mean()) if len(d) else np.nan,
        "absolute_ci95_lo": a_lo,
        "absolute_ci95_hi": a_hi,
        "median_absolute_shift_pp": float(np.median(abs_d)) if len(d) else np.nan,
        "share_low_price_preference": float(np.mean(d > 0)) if len(d) else np.nan,
        "share_high_price_preference": float(np.mean(d < 0)) if len(d) else np.nan,
        "share_exactly_invariant": float(np.mean(d == 0)) if len(d) else np.nan,
        "p_paired_t": paired_t_p(d),
        "p_wilcoxon": wilcoxon_p(d),
        "p_sign_flip": sign_flip_p(d, seed=ANALYSIS_SEED + 2000 + seed_offset),
        "p_equivalence_2pp": tost_p(d),
    }


def _add_normative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = df["weight_a"].to_numpy() / 100.0
    wstar = df["optimal_weight_a"].to_numpy()
    args = [df[c].to_numpy() for c in ["mu_a", "mu_b", "sigma_a", "sigma_b", "rho", "gamma"]]
    df["utility_model"] = utility(w, *args)
    df["utility_optimum"] = utility(wstar, *args)
    df["utility_regret_bps"] = np.maximum(
        0.0, (df["utility_optimum"] - df["utility_model"]) * 10_000
    )
    df["abs_weight_error_pp"] = np.abs(w - wstar) * 100
    return df


def analyze(path: Path, outdir: Path) -> None:
    df = pd.read_csv(path)
    required = {
        "scenario_id", "scenario_type", "condition_id", "model", "weight_a",
        "optimal_weight_a", "mu_a", "mu_b", "sigma_a", "sigma_b", "rho", "gamma",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    outdir.mkdir(parents=True, exist_ok=True)
    df = _add_normative_metrics(df)
    df.to_csv(outdir / "analysis_dataset.csv", index=False)

    treatment_rows: list[dict] = []
    paired_rows: list[dict] = []
    noise_by_model: dict[str, pd.Series] = {}

    for model, gm in df.groupby("model", sort=True):
        neutral = gm[gm.condition_id.isin(["neutral_r1", "neutral_r2"])].pivot(
            index="scenario_id", columns="condition_id", values="weight_a"
        ).dropna()
        if {"neutral_r1", "neutral_r2"}.issubset(neutral.columns):
            baseline_d = (neutral["neutral_r1"] - neutral["neutral_r2"]).dropna()
        else:
            baseline_d = pd.Series(dtype=float)
        baseline_abs = baseline_d.abs()
        noise_by_model[model] = baseline_abs
        row = {"model": model, "scenario_type": "all", "ratio": 1, "baseline_repeat_noise": True}
        row.update(_effect_summary(baseline_d.to_numpy(), seed_offset=1))
        row["mean_excess_abs_shift_vs_repeat_pp"] = 0.0
        row["p_excess_abs_shift_vs_repeat"] = np.nan
        treatment_rows.append(row)

        strata = [("all", gm)]
        strata.extend((t, gm[gm.scenario_type == t]) for t in sorted(gm.scenario_type.unique()))
        for stratum, gs in strata:
            for ratio, a_id, b_id in PAIRS:
                p = gs[gs.condition_id.isin([a_id, b_id])].pivot(
                    index="scenario_id", columns="condition_id", values="weight_a"
                ).dropna()
                if not {a_id, b_id}.issubset(p.columns):
                    continue
                d = (p[a_id] - p[b_id]).dropna()
                if d.empty:
                    continue
                summary = {
                    "model": model,
                    "scenario_type": stratum,
                    "ratio": ratio,
                    "baseline_repeat_noise": False,
                }
                summary.update(_effect_summary(d.to_numpy(), seed_offset=ratio + len(model) + len(stratum)))

                common = d.index.intersection(baseline_abs.index)
                excess = d.loc[common].abs() - baseline_abs.loc[common]
                ex_lo, ex_hi = mean_ci(excess.to_numpy(), seed=ANALYSIS_SEED + ratio + 4000)
                summary.update({
                    "mean_excess_abs_shift_vs_repeat_pp": float(excess.mean()) if len(excess) else np.nan,
                    "excess_abs_ci95_lo": ex_lo,
                    "excess_abs_ci95_hi": ex_hi,
                    "p_excess_abs_shift_vs_repeat": paired_t_p(excess.to_numpy()),
                    "p_excess_abs_wilcoxon": wilcoxon_p(excess.to_numpy()),
                })
                treatment_rows.append(summary)

                for sid, value in d.items():
                    scenario_type = gs.loc[gs.scenario_id == sid, "scenario_type"].iloc[0]
                    paired_rows.append({
                        "scenario_id": sid,
                        "model": model,
                        "scenario_type": scenario_type,
                        "ratio": ratio,
                        "directional_effect_pp": float(value),
                        "absolute_shift_pp": float(abs(value)),
                        "neutral_repeat_abs_pp": float(baseline_abs.get(sid, np.nan)),
                    })

    treatment = pd.DataFrame(treatment_rows)
    treatment.to_csv(outdir / "treatment_summary.csv", index=False)
    paired = pd.DataFrame(paired_rows).drop_duplicates(["scenario_id", "model", "ratio"])
    paired.to_csv(outdir / "paired_effects.csv", index=False)

    dose_rows = []
    if not paired.empty:
        wide = paired.pivot(index=["scenario_id", "model"], columns="ratio", values="absolute_shift_pp").dropna()
        for model in sorted(wide.index.get_level_values("model").unique()):
            wm = wide.xs(model, level="model")
            for hi, lo in [(10, 2), (50, 10), (50, 2)]:
                delta = wm[hi] - wm[lo]
                ci_lo, ci_hi = mean_ci(delta, seed=ANALYSIS_SEED + hi * 10 + lo)
                dose_rows.append({
                    "model": model,
                    "higher_ratio": hi,
                    "lower_ratio": lo,
                    "n": len(delta),
                    "mean_change_abs_shift_pp": float(delta.mean()),
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                    "p_paired_t": paired_t_p(delta),
                    "p_wilcoxon": wilcoxon_p(delta),
                })
    pd.DataFrame(dose_rows).to_csv(outdir / "dose_response.csv", index=False)

    welfare_rows = []
    for model, gm in df.groupby("model", sort=True):
        neutral = gm[gm.condition_id.isin(["neutral_r1", "neutral_r2"])].groupby("scenario_id").agg(
            neutral_regret_bps=("utility_regret_bps", "mean"),
            neutral_error_pp=("abs_weight_error_pp", "mean"),
        )
        for ratio, a_id, b_id in PAIRS:
            treated = gm[gm.condition_id.isin([a_id, b_id])].groupby("scenario_id").agg(
                treated_regret_bps=("utility_regret_bps", "mean"),
                treated_error_pp=("abs_weight_error_pp", "mean"),
            )
            joined = neutral.join(treated, how="inner")
            regret_delta = joined.treated_regret_bps - joined.neutral_regret_bps
            error_delta = joined.treated_error_pp - joined.neutral_error_pp
            rlo, rhi = mean_ci(regret_delta, seed=ANALYSIS_SEED + 5000 + ratio)
            elo, ehi = mean_ci(error_delta, seed=ANALYSIS_SEED + 6000 + ratio)
            welfare_rows.append({
                "model": model,
                "ratio": ratio,
                "n": len(joined),
                "mean_regret_increase_bps": float(regret_delta.mean()),
                "regret_ci95_lo": rlo,
                "regret_ci95_hi": rhi,
                "p_regret_increase_paired_t": paired_t_p(regret_delta),
                "mean_abs_error_increase_pp": float(error_delta.mean()),
                "error_ci95_lo": elo,
                "error_ci95_hi": ehi,
                "p_error_increase_paired_t": paired_t_p(error_delta),
            })
    pd.DataFrame(welfare_rows).to_csv(outdir / "welfare_effects.csv", index=False)

    descriptive = df.groupby(["model", "condition_id"], as_index=False).agg(
        n=("scenario_id", "size"),
        mean_abs_weight_error_pp=("abs_weight_error_pp", "mean"),
        mean_utility_regret_bps=("utility_regret_bps", "mean"),
        median_utility_regret_bps=("utility_regret_bps", "median"),
    )
    descriptive.to_csv(outdir / "welfare_descriptive.csv", index=False)

    print(treatment[treatment.scenario_type == "all"].to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--outdir", type=Path, default=Path("results/analysis"))
    args = p.parse_args()
    analyze(args.results, args.outdir)


if __name__ == "__main__":
    main()
