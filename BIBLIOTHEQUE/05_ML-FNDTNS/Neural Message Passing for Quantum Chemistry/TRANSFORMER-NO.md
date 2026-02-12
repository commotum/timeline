# Neural Message Passing for Quantum Chemistry (2017)
Source: Neural Message Passing for Quantum Chemistry.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract centers the method on Message Passing Neural Networks (MPNNs), not Transformer blocks or self-attention mechanisms.
- Auxiliary analyses characterize the model behavior as message passing with static neighbor aggregation rather than Transformer-style self-attention.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "In this paper, we reformulate existing models into a single common framework we call Message Passing Neural Networks (MPNNs) and explore additional novel variations within this framework." (Abstract, `Neural Message Passing for Quantum Chemistry.md`)
- "the model constructs internal node states via message passing with fixed neighbor aggregation (constructed state and static attention, inferred)." (Summary, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a non-Transformer central architecture (MPNN/message passing; no self-attention core indicated).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence evidence.
