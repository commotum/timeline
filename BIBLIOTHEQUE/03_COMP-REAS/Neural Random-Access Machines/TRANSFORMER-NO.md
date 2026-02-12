# Neural Random-Access Machines (Year not specified)
Source: Neural Random-Access Machines.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines the core architecture as Neural Random-Access Machine for pointer manipulation over external random-access memory, with no Transformer-style self-attention as a central component.
- The auxiliary analyses describe pointer-based READ/WRITE memory operations and an LSTM/MLP controller rather than Transformer-family blocks.
- The Extending-dimensions analysis markdown was unavailable (resolved as `MISSING`), so the decision uses the abstract and the three available auxiliary files.

## Evidence
- "In this paper, we propose and investigate a new neural network architecture called Neural Random Access Machine. It can manipulate and dereference pointers to an external variable-size random-access memory." (Abstract, Neural Random-Access Machines.md)
- "The LSTM controller gets the \"binarized\" values  $r_1, r_2, \ldots$  stored in the registers as inputs and outputs the description of the circuit in the grey box..." (Evidence quote recorded in TASK-DOMAINS.md, Figure 2 caption)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence TRANSFORMER-NO using abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient to finalize.
