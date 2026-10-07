# Preregistered experimental design

## Research question

Does an economically irrelevant change in nominal share price alter portfolio allocations produced by large language models?

A stock split changes the nominal price per share but, absent trading frictions, does not change the economic opportunity set. The experiment therefore treats nominal share price assignment as a randomized, payoff-irrelevant treatment.

## Hypotheses

**H1, non-invariance.** Holding expected returns, volatilities, return correlation, objective, investable wealth, and all trading constraints fixed, swapping which security receives the lower nominal share price changes the model's allocation.

**H2, directional low-price preference.** The allocation to Stock A is higher when A is assigned the lower nominal price than when B is assigned the lower nominal price.

**H3, dose response.** The absolute allocation shift increases with the nominal price ratio, ordered 2x, 10x, 50x.

**H4, financial consequence.** Nominal-price treatments increase deviation from the exact mean-variance optimum and increase mean-variance utility regret relative to the neutral-price condition.

## Experimental units

The locked design contains 300 financial states generated before any model outcomes are observed.

* 100 symmetric states. Stocks A and B have identical expected returns and volatilities. The exact optimal allocation is 50/50.
* 200 asymmetric states. Expected returns, volatilities, and correlations vary. States are retained only when the exact long-only mean-variance optimum for Stock A lies between 15% and 85%.

The random seed is `20261007`. The deterministic generator and the SHA-256 hash of the resulting 300-state design are committed before live inference. The workflow materializes the exact CSV from that frozen generator before any model call.

## Conditions

Every state is presented in eight conditions.

1. `neutral_r1`: equal nominal prices.
2. `neutral_r2`: exact duplicate of `neutral_r1`, used to estimate technical/model nondeterminism.
3. `2x_Acheap`: A receives the low price and B the 2x price.
4. `2x_Bcheap`: B receives the low price and A the 2x price.
5. `10x_Acheap`.
6. `10x_Bcheap`.
7. `50x_Acheap`.
8. `50x_Bcheap`.

Within each A-cheap/B-cheap pair, every prompt element other than the assignment of the two nominal prices is identical. The display order of A and B is randomized once at the state level and then held fixed across all conditions in that state.

## Portfolio problem

Each model allocates a $10,000 long-only portfolio between two fictional equities. Fractional shares are explicitly available. Fees, taxes, minimum lots, shorting, leverage, and liquidity differences are excluded.

The objective is fixed as

`U = E[R_p] - gamma/2 * Var(R_p)`

with `gamma = 3`.

The model returns integer percentage weights summing to 100. No explanation is requested.

## Models

Four independently developed model families are used:

* OpenAI GPT-5.6 Luna
* DeepSeek V4 Flash
* Qwen3.5 397B A17B
* Meta Llama 4 Maverick

Exact OpenRouter slugs and the price snapshot are stored in `config/models.json`.

## Inference settings

Temperature is set to 0 and maximum completion length is 32 tokens. The prompt requires a two-field JSON answer and no explanation. Each model is pinned to one serving provider and fallbacks are disabled. GPT-5.6 Luna, DeepSeek V4 Flash, and Qwen3.5 397B A17B use explicit `none` reasoning effort; Llama 4 Maverick receives no reasoning parameter. These four profiles repeatedly produced valid terse allocations in the engineering pilots. Responses are requested independently, with no conversation history shared across observations. The complete call schedule is deterministically shuffled with request-order seed `20261008` before dispatch so treatment condition is not aligned with collection time. The serving model, provider metadata when available, token usage, raw response, and parse status are retained.

## Primary estimands

For state `s`, model `m`, and price ratio `r`, define

`D_smr = w_A(A cheap) - w_A(B cheap)`.

The directional estimand is the mean of `D`. A positive value indicates preference for the nominally cheaper security.

The primary non-invariance estimand is `mean(abs(D))`. This is compared with the identical-prompt repeat noise from `abs(w_A(neutral_r1) - w_A(neutral_r2))`.

## Financial consequence measures

For every state the exact optimal weight `w*` is calculated analytically. Outcomes include:

* absolute allocation error, in percentage points
* mean-variance utility regret, converted to annualized basis-point units
* treatment-induced change in allocation error
* treatment-induced change in utility regret

## Statistical analysis

Inference is paired within economic state and model. The main analyses report model-specific and pooled effects, cluster-bootstrap confidence intervals by underlying state, paired t-tests, Wilcoxon signed-rank tests, and randomization-based inference where appropriate.

The dose-response analysis tests whether absolute shifts increase from 2x to 10x to 50x.

A practical equivalence region of +/-2 percentage points is reserved for a null/invariance interpretation. The exact neutral duplicate distribution is reported alongside all treatment effects.

## Exclusions

A response is excluded from numerical analysis only when it cannot be parsed into two valid weights between 0 and 100 that sum to 100 after the predefined parser and retry policy. All such failures are retained and reported by model and condition. No financial state is removed after outcomes are observed.

## Budget rule

The user-set hard OpenRouter budget is $1.35. The collection code uses a conservative live-run stop at $1.20 and reads the dedicated API key's usage meter during collection. The experiment stops scheduling new calls when the internal guard reaches the stop level. The final manifest records estimated and key-meter spend.

## Analysis freeze

This file, the scenario generator, model configuration, random seed, and expected design hash are committed before the full live run. Any later exploratory analysis is to be labeled exploratory rather than preregistered.

### Pre-confirmatory model compatibility amendment

No confirmatory observations had been collected when the engineering pilots established that GLM 5.3 Flash could not reliably emit a final portfolio allocation inside the fixed 32-token completion budget across tested serving routes. Its endpoints consumed the completion budget in reasoning before returning the requested JSON allocation. This is a technical incompatibility with the locked low-cost response protocol, not an observed treatment result.

GLM 5.3 Flash is therefore replaced before the confirmatory run by Google Gemini 3.8 Flash. The final five model families are OpenAI, DeepSeek, Google, Qwen, and Meta. The 300 financial states, eight conditions, random seeds, hypotheses, estimands, exclusion rules, equivalence margin, and statistical analysis plan are unchanged. Engineering-pilot observations remain excluded from confirmatory inference.

### Final confirmatory model-set freeze

A subsequent engineering probe showed that Gemini 3.8 Flash on the tested Google endpoint requires reasoning and rejects a non-reasoning request. Because the experiment deliberately fixes a terse 32-token completion budget to keep all models on a comparable low-latency decision protocol, the Google candidate is not added to the confirmatory set.

The final confirmatory model set therefore contains the four families that repeatedly completed the engineering protocol without model-specific relaxation: OpenAI GPT-5.6 Luna, DeepSeek V4 Flash, Qwen3.5 397B A17B, and Meta Llama 4 Maverick.

This gives 2,400 decisions per model and 9,600 confirmatory decisions in total. The scenario count, treatment conditions, hypotheses, estimands, statistical tests, equivalence threshold, and randomization scheme remain unchanged. No engineering-pilot observation is included in confirmatory inference.
