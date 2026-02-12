# Learning Modular Exponentiation with Transformers (Year not specified)
Source: Learning Modular Exponentiation with Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly states the central model is a "4-layer encoder—decoder Transformers" trained for the main task, so self-attention is part of the core method rather than a peripheral baseline.
- Auxiliary analyses (`TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, `TASK_MODEL_RATIO.md`) consistently describe one modular-exponentiation prediction task solved with transformer models; the Extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "We train compact 4-layer encoder—decoder Transformers to predict d and analyze how they come to solve the task." (Abstract, `Learning Modular Exponentiation with Transformers.md`)
- "Causal analysis shows that, on instances without reduction  $(c > a^b)$ , a small circuit consisting only of final-layer attention heads reproduces full-model behavior..." (Abstract, `Learning Modular Exponentiation with Transformers.md`)
- "We train compact 4-layer encoder—decoder Transformers to predict d" (Evidence section, `TASK-DOMAINS.md`)
- "In this work, we train transformer models to perform modular exponentiation..." (`TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer-central classification; Extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Not needed because Pass 1 already established a clear Transformer-based central model.
