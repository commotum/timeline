# MemoryLLM: Plug-n-Play Interpretable Feed-Forward Memory for Transformers (2026)
Source: MemoryLLM- Plug-n-Play Interpretable Feed-Forward Memory for Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly frames MemoryLLM as a modification of Transformer blocks and repeatedly centers self-attention as a core module.
- Auxiliary analyses (`TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, `TASK_MODEL_RATIO.md`) are consistent with a single Transformer-family LLM model evaluated across tasks; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "MemoryLLM, which aims to decouple FFNs from self-attention" (Abstract, `MemoryLLM- Plug-n-Play Interpretable Feed-Forward Memory for Transformers.md`)
- "We also introduce Flex-MemoryLLM, positioning it between a conventional transformer design and MemoryLLM." (Abstract, `MemoryLLM- Plug-n-Play Interpretable Feed-Forward Memory for Transformers.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-YES from abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for a high-confidence decision.
