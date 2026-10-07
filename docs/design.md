# Design notes

The experiment is intentionally synthetic rather than market-backtest based. This removes contemporaneous news, data leakage, look-ahead bias, changing fundamentals, and disagreement about the information set. Nominal price is randomized while the economic state is held constant.

The strongest identification comes from within-state counterfactual pairs. A given model sees the same expected returns, volatilities, correlation, objective, portfolio size, display order, and constraints in both prompts. Only the nominal prices attached to labels A and B are swapped.

Two exact neutral-price repetitions provide a direct empirical baseline for residual nondeterminism at temperature zero. This allows treatment shifts to be interpreted relative to ordinary same-prompt variation rather than assuming perfect determinism.

The symmetric block provides an especially transparent benchmark because the exact optimum is 50/50. The asymmetric block establishes external relevance beyond trivial equal-asset choices and permits welfare calculations against an exact state-specific optimum.
