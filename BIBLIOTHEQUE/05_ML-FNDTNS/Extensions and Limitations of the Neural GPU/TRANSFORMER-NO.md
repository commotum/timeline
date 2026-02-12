# Extensions and Limitations of the Neural GPU (Year not specified)
Source: Extensions and Limitations of the Neural GPU.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analysis describe the central model as a Neural GPU using convolutional recurrent computation for arithmetic tasks, not Transformer-style self-attention blocks.
- Auxiliary indicators describe attention/state as static/constructed rather than self-attention-driven, and the Extending-dimensions analysis markdown was unavailable (`MISSING`).

## Evidence
- "The Neural GPU is a recent model that can learn algorithms such as multi-digit binary addition and binary multiplication in a way that generalizes to inputs of arbitrary length." (Abstract, `Extensions and Limitations of the Neural GPU.md`)
- "The Neural GPU architecture is the combination of a convolution on variable size inputs with a recurrent neural network" (Section 3 Model quote recorded in `TASK-DOMAINS.md`)
- "\"Binary multi-digit addition\",\"Binary digit sequences (two integers)\",\"1D (t) (inferred)\",\"Open\",\"Static (inferred)\",\"Constructed (inferred)\",\"Binary digit sequence (sum)\",\"1D (t) (inferred)\",\"Open\"" (Row 2, `TASK-DOMAINS.csv`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence decision.
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
