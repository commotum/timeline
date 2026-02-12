# Move Evaluation in Go Using Deep Convolutional Neural Networks (Year not specified)
Source: Move Evaluation in Go Using Deep Convolutional Neural Networks.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The central architecture is a deep convolutional neural network (12-layer CNN), not a Transformer or self-attention model.
- Auxiliary analyses also characterize the model as CNN-based with static (non-self-attention) dynamics.
- `EXTENDING-DIMENSIONS.md` was unavailable (`MISSING`) and was skipped per instructions.

## Evidence
- "We train a large 12-layer convolutional neural network by supervised learning from a database of human professional games." (Move Evaluation in Go Using Deep Convolutional Neural Networks.md, Abstract)
- "The paper trains a CNN to predict expert Go moves from 19x19 board feature planes and outputs a distribution over 361 board positions." (TASK-DOMAINS.md, Summary)
- "\"Move prediction (expert next-move classification)\",\"Go board position feature planes (19x19)\",\"2D (x, y) (inferred)\",\"Fixed (inferred)\",\"Static (inferred)\",\"Direct (inferred)\"" (TASK-DOMAINS.csv, row 1)
- "When the trained convolutional network was used directly to play games of Go, without any search..." (TASK_MODEL_RATIO.md, evidence quote from ABSTRACT)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO decision from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; EXTENDING-DIMENSIONS.md unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
