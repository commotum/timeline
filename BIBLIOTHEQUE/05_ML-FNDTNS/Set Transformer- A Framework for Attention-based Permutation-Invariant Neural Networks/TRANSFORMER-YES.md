# Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks (2019)
Source: Set Transformer- A Framework for Attention-based Permutation-Invariant Neural Networks.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines the proposed method as an attention-based Set Transformer and states that both encoder and decoder rely on attention mechanisms.
- The paper’s core contribution is self-attention for set modeling (including induced attention to scale self-attention), and auxiliary analyses consistently treat attention as central rather than peripheral.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), but remaining Pass 1 sources were sufficient for a high-confidence decision.

## Evidence
- "We present an attention-based neural network module, the Set Transformer, specifically designed to model interactions among elements in the input set." (Abstract, `Set Transformer- A Framework for Attention-based Permutation-Invariant Neural Networks.md`)
- "The model consists of an encoder and a decoder, both of which rely on attention mechanisms." (Abstract, `Set Transformer- A Framework for Attention-based Permutation-Invariant Neural Networks.md`)
- "a Set Transformer consists of an encoder followed by a decoder, but a distinguishing feature is that each layer in the encoder and decoder attends to their inputs" (`TASK-DOMAINS.md`, Evidence section inference note citing Section 3)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - high-confidence TRANSFORMER-YES from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient for a high-confidence decision.
