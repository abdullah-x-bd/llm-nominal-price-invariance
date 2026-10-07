# LLM Nominal Price Invariance

Controlled experiments on whether nominal share prices, which are payoff-irrelevant when fractional trading is available and frictions are absent, alter portfolio allocations produced by large language models.

## Core design

The locked experiment contains 300 financial states, eight nominal-price conditions, and four independently developed model families. Each model therefore makes 2,400 independent portfolio decisions, for 9,600 decisions in the confirmatory experiment.

The treatment is deliberately narrow. Within each paired condition, the economic state and prompt are unchanged except for which stock label receives the lower nominal share price.

See `docs/preregistration.md` for the frozen hypotheses and analysis plan.

## Models

* `openai/gpt-5.6-luna`
* `deepseek/deepseek-v4-flash`
* `qwen/qwen3.5-397b-a17b`
* `meta-llama/llama-4-maverick`

The OpenRouter pricing snapshot used for budget planning is in `config/models.json`.

## Reproduce the design

```bash
python -m pip install -e '.[dev]'
python scripts/generate_design.py
pytest
```

The design generator is deterministic. Re-running it should reproduce the committed SHA-256 hash in `data/design/manifest.json`.

## Run inference

Set an OpenRouter key locally as `OPENROUTER_API_KEY`, or add that name as a GitHub Actions repository secret.

Pilot:

```bash
python -m nominal_price_invariance.collect --mode pilot
```

Full collection:

```bash
python -m nominal_price_invariance.collect --mode full
```

The collection code has a $1.20 live-run stop below the $1.35 hard budget. It also queries the dedicated key's usage meter during collection and records a spend manifest.

## Analyze

```bash
python -m nominal_price_invariance.analysis data/results/results.csv
```

The analysis reports paired nominal-price effects, exact-repeat noise, allocation error relative to the analytic optimum, and mean-variance utility regret.

## Repository structure

```text
config/                 model slugs and price snapshot
data/design/            locked experimental states and manifest
docs/                   preregistration and design notes
scripts/                design generation and ex-ante power checks
src/                     collection and analysis package
tests/                   deterministic design and parser tests
.github/workflows/       CI and manual inference workflows
```

## Status

Experimental design and code are frozen before the full inference run. Live results should not be used to alter the preregistered hypotheses or state-generation rule.
