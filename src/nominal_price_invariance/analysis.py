from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PAIRS = [(2, "2x_Acheap", "2x_Bcheap"), (10, "10x_Acheap", "10x_Bcheap"), (50, "50x_Acheap", "50x_Bcheap")]


def utility(w: pd.Series | np.ndarray, mu_a, mu_b, sigma_a, sigma_b, rho, gamma):
    w = np.asarray(w, dtype=float)
    mu = w * np.asarray(mu_a) + (1 - w) * np.asarray(mu_b)
    cov = np.asarray(rho) * np.asarray(sigma_a) * np.asarray(sigma_b)
    var = (
        w**2 * np.asarray(sigma_a)**2
        + (1 - w)**2 * np.asarray(sigma_b)**2
        + 2 * w * (1 - w) * cov
    )
    return mu - 0.5 * np.asarray(gamma) * var


def analyze(path: Path, outdir: Path) -> None:
    df = pd.read_csv(path)
    outdir.mkdir(parents=True, exist_ok=True)

    w = df["weight_a"].to_numpy() / 100.0
    wstar = df["optimal_weight_a"].to_numpy()
    args = [df[c].to_numpy() for c in ["mu_a", "mu_b", "sigma_a", "sigma_b", "rho", "gamma"]]
    df["utility_model"] = utility(w, *args)
    df["utility_optimum"] = utility(wstar, *args)
    df["utility_regret_bps"] = np.maximum(0.0, (df["utility_optimum"] - df["utility_model"]) * 10_000)
    df["abs_weight_error_pp"] = np.abs(w - wstar) * 100

    treatment_rows = []
    for model, gm in df.groupby("model"):
        neutral = gm[gm.condition_id.isin(["neutral_r1", "neutral_r2"])].pivot(
            index="scenario_id", columns="condition_id", values="weight_a"
        ).dropna()
        baseline_d = neutral["neutral_r1"] - neutral["neutral_r2"]
        treatment_rows.append({
            "model": model, "ratio": 1, "n": len(neutral),
            "mean_directional_effect_pp": baseline_d.mean(),
            "mean_absolute_shift_pp": np.abs(baseline_d).mean(),
            "median_absolute_shift_pp": np.median(np.abs(baseline_d)),
            "p_directional": np.nan,
            "baseline_repeat_noise": True,
        })

        for ratio, a_id, b_id in PAIRS:
            p = gm[gm.condition_id.isin([a_id, b_id])].pivot(
                index="scenario_id", columns="condition_id", values="weight_a"
            ).dropna()
            d = p[a_id] - p[b_id]
            t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
            treatment_rows.append({
                "model": model, "ratio": ratio, "n": len(d),
                "mean_directional_effect_pp": d.mean(),
                "mean_absolute_shift_pp": np.abs(d).mean(),
                "median_absolute_shift_pp": np.median(np.abs(d)),
                "p_directional": float(t.pvalue) if t is not None else np.nan,
                "baseline_repeat_noise": False,
            })

    treatment = pd.DataFrame(treatment_rows)
    treatment.to_csv(outdir / "treatment_summary.csv", index=False)

    welfare = df.groupby(["model", "condition_id"], as_index=False).agg(
        n=("scenario_id", "size"),
        mean_abs_weight_error_pp=("abs_weight_error_pp", "mean"),
        mean_utility_regret_bps=("utility_regret_bps", "mean"),
        median_utility_regret_bps=("utility_regret_bps", "median"),
    )
    welfare.to_csv(outdir / "welfare_summary.csv", index=False)

    paired = []
    for ratio, a_id, b_id in PAIRS:
        x = df[df.condition_id.isin([a_id, b_id])].pivot_table(
            index=["scenario_id", "model"], columns="condition_id", values="weight_a", aggfunc="first"
        ).dropna()
        for (sid, model), row in x.iterrows():
            paired.append({"scenario_id": sid, "model": model, "ratio": ratio,
                           "directional_effect_pp": row[a_id] - row[b_id],
                           "absolute_shift_pp": abs(row[a_id] - row[b_id])})
    pd.DataFrame(paired).to_csv(outdir / "paired_effects.csv", index=False)

    print(treatment.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--outdir", type=Path, default=Path("results/analysis"))
    args = p.parse_args()
    analyze(args.results, args.outdir)


if __name__ == "__main__":
    main()
