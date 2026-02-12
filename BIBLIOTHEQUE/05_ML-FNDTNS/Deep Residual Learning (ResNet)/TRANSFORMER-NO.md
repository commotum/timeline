# Deep Residual Learning for Image Recognition (2015)
Source: Deep Residual Learning (ResNet).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes deep residual convolutional networks with identity/shortcut-style residual learning, not Transformer-style self-attention blocks.
- Auxiliary analyses do not indicate self-attention as a core mechanism, and the extending-dimensions file was unavailable (`MISSING`) but not necessary for a confident decision.

## Evidence
- "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously." (Deep Residual Learning (ResNet).md, Abstract)
- "Dynamics are described as fixed or capped where explicit (e.g., fixed-size crops or a fixed set of image scales and proposal counts), while attention and state dynamics are largely not specified." (TASK-DOMAINS.md, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO decision from the abstract plus TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; extending-dimensions analysis file was unavailable (`MISSING`).
