# Neural Programmer-Interpreters (NPI) (Year not specified)
Source: Neural Programmer-Interpreters (NPI).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines NPI as a recurrent architecture with a recurrent core/program memory setup, and does not present Transformer-style self-attention as part of the central model.
- Auxiliary analyses are consistent with a recurrent/LSTM-centric model-family signal and provide no evidence of Transformer blocks; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "We propose the neural programmer-interpreter (NPI): a recurrent and compositional neural network that learns to represent and execute programs." (Abstract, `Neural Programmer-Interpreters (NPI).md`)
- "NPI has three learnable components: a task-agnostic recurrent core, a persistent key-value program memory, and domain-specific encoders..." (Abstract, `Neural Programmer-Interpreters (NPI).md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence NON-transformer classification from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
