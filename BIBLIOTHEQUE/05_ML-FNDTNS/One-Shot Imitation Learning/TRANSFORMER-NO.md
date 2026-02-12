# One-Shot Imitation Learning (Year not specified)
Source: One-Shot Imitation Learning.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract reports "soft attention" for generalization, but does not describe Transformer-style self-attention blocks as the core model architecture.
- The auxiliary analysis indicates LSTM-based attention behavior and no Transformer-family model cues; `EXTENDING-DIMENSIONS.md` was unavailable.

## Evidence
- "Our experiments show that the use of soft attention allows the model to generalize to conditions and tasks unseen in the training data." (Abstract, `One-Shot Imitation Learning.md`)
- "the LSTM outputs a weighting over the different landmarks from the demonstration sequence." (Appendix A quote recorded in `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence non-Transformer decision from the abstract and available auxiliary files; extending-dimensions analysis was unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 was already decisive.
