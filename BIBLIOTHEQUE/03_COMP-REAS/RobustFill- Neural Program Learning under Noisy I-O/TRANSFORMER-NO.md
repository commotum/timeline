# RobustFill: Neural Program Learning under Noisy I/O (2017)
Source: RobustFill- Neural Program Learning under Noisy I-O.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly identifies the core architecture as a modified attention RNN, not a Transformer/self-attention block architecture.
- Auxiliary analysis describes LSTM-based encoding/decoding with double attention; attention is present, but within recurrent models rather than Transformer-style self-attention layers.

## Evidence
- "Our neural models use a modified attention RNN to allow encoding of variable-sized sets of I/O pairs." (Abstract, `RobustFill- Neural Program Learning under Noisy I-O.md`)
- "There is an additional LSTM to encode  $I^y$ . The decoder layer  $O^y$  uses double attention on  $O_i$  and  $I^y$ ." (`TASK-DOMAINS.md`, Evidence section, Program induction)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence NO decision; `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md` are consistent with recurrent attentional models, and Extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided clear architecture signals.
