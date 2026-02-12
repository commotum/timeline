# NEURAL GPUS LEARN ALGORITHMS (Year not specified)
Source: Neural GPUs Learn Algorithms.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes the core model as a convolutional gated recurrent architecture (Neural GPU), not a self-attention/Transformer architecture.
- Auxiliary analyses consistently label attention behavior as static/inferred and do not indicate Transformer blocks as central; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "We present a neural network architecture to address this problem: the *Neural GPU*. It is based on a type of convolutional gated recurrent unit and, like the NTM, is computationally universal." (Abstract, `Neural GPUs Learn Algorithms.md`)
- "The text emphasizes handling inputs of arbitrary size, so input/output dynamics are treated as open, while the model processes fixed inputs without runtime attention selection and evolves an internal recurrent state (static attention and constructed state, inferred from the architecture description)." (Summary, `TASK-DOMAINS.md`)
- "Long binary multiplication,\"Binary digit sequence for two lower-endian numbers with a separator symbol (PAD possible)\",\"1D (t)\",\"Open (inferred)\",\"Static (inferred)\",\"Constructed (inferred)\"" (Row excerpt, `TASK-DOMAINS.csv`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence NO decision; `MISSING` extending-dimensions analysis was unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient to finalize.
