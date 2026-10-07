# Engineering pilot protocol

The full experimental hypotheses, state generator, treatment conditions, and estimands were frozen before any live model call. A two-state engineering pilot was then run to validate API compatibility and cost accounting. Pilot observations are not part of the confirmatory sample and are not used to alter hypotheses or financial-state generation.

## Pilot 1

The first pilot scheduled 80 calls: two financial states, eight conditions, and five model families.

Three models completed all 16 calls: GPT-5.6 Luna, DeepSeek V4 Flash, and Qwen3.5 397B A17B. GLM 5.3 Flash returned request-level HTTP 400 errors under the generic reasoning configuration. Llama 4 Maverick completed 4 of 16 calls. Its successful calls were served by Novita and returned the requested JSON, while another serving route produced explanatory text until the 32-token cap rather than the requested JSON.

OpenRouter's dedicated key meter recorded $0.002472 of spend for Pilot 1.

## Pre-full-run compatibility amendment

No hypothesis, state, treatment, outcome, or estimand was changed. The transport layer was amended before the full sample:

1. One serving provider is pinned for each model to prevent serving-provider variation from becoming an uncontrolled source of model variation.
2. Fallbacks are disabled for the confirmatory run.
3. `response_format` requests JSON output and `require_parameters` restricts requests to endpoints that support the requested parameters.
4. The generic reasoning setting is removed from models for which it is not part of the locked request profile. GPT-5.6 Luna retains explicit `none` reasoning effort.
5. Non-success HTTP response bodies are retained in the retry audit trail.
6. Analysis code skips incomplete pilot pairs rather than treating missing conditions as zero effects.

A second engineering pilot is required to pass all 80 calls before the full 12,000-decision collection is started.

## Pilot 2

Pilot 2 validated fixed-provider routing but showed that a universal structured-output request profile was inappropriate across these model families. Llama 4 Maverick completed 16/16 calls when pinned to Novita. Qwen3.5 397B A17B completed 16/16 calls but, without its explicit non-reasoning control, generated a large volume of reasoning tokens and was economically unsuitable for the confirmatory budget. DeepSeek V4 Flash similarly consumed the 32-token cap in reasoning rather than returning the requested allocation. The pinned OpenAI route for GPT-5.6 Luna rejected the universal parameter set.

The dedicated key meter recorded $0.041355 of incremental spend for Pilot 2. These observations are used only to select a technically compatible request profile.

## Final compatibility profile proposed for Pilot 3

The prompt-level JSON instruction is retained for all models. Universal `response_format` and `require_parameters` are removed. Provider pins and disabled fallbacks are retained. Explicit `none` reasoning is restored for GPT-5.6 Luna, DeepSeek V4 Flash, and Qwen3.5 397B A17B because Pilot 1 demonstrated clean one-pass JSON responses at low cost under that setting. Llama 4 Maverick remains pinned to Novita without a reasoning parameter. GLM 5.3 Flash remains pinned to DeepInfra with no reasoning parameter.

No confirmatory outcome, treatment, financial state, or estimand is changed. Pilot 3 must complete all 80 engineering calls without malformed outputs before the full sample can begin.
