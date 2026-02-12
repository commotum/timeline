# Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (2024)
Source: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s main method is built around LLM inference-time scaling (proposal/revision/search/verifier), and the named base model family is PaLM 2, which is a Transformer-family LLM.
- The auxiliary task/model analysis shows one central model family (PaLM 2-S*) used for the core results, so Transformer-style self-attention is part of the primary model, not just a baseline mention.

## Evidence
- "Enabling LLMs to improve their outputs by using more test-time computation is a critical step towards building generally self-improving agents..." (abstract in `Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.md`)
- "We conduct our analysis using the PaLM 2-S* [3] (Codey) base model." (`TASK_MODEL_RATIO.md`, quoted from Section 4, Models)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for a high-confidence decision.
