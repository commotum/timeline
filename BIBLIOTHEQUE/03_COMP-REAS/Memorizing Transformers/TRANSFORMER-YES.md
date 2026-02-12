# MEMORIZING TRANSFORMERS (Year not specified)
Source: Memorizing Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes adding memory via retrieval over stored representations at inference time, aligning with Transformer-style attention behavior.
- The auxiliary analysis explicitly indicates dynamic attention with per-query retrieval from external `(key, value)` memory, consistent with a Transformer attention extension.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "In this work, we extend language models with the ability to memorize the internal representations of past inputs." (Abstract, `Memorizing Transformers.md`)
- "approximate k-nearest-neighbor search into the external memory" and "contain a different set of (key, value) pairs for each query" (Evidence section, `TASK-DOMAINS.md`, citing Section 3.1)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-YES from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`)
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient to finalize
