# Solving Rubik's Cube with a Robot Hand (2019)
Source: Solving Rubik's Cube with a Robot Hand.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes ADR-trained control and vision models with memory augmentation, but does not indicate Transformer or self-attention blocks as central architecture.
- Auxiliary analysis explicitly points to recurrent LSTM policy and CNN/fully connected vision pipeline, and marks attention dynamics as static/inferred; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "For control policies, memory-augmented models trained on an ADR-generated distribution of environments show clear signs of emergent meta-learning at test time." (Solving Rubik's Cube with a Robot Hand.md, Abstract, line 17)
- "The policy is still recurrent since only a policy with access to some form of memory can perform meta-learning. We still use a single feed-forward layer with a ReLU activation [72] followed by a single LSTM layer [45]." (TASK-DOMAINS.md, line 20)
- "These three feature maps are then flattened, concatenated, and fed into a stack of fully-connected layers which ultimately produce predictions sufficient for tracking the full state of the cube, including the position, orientation, and face angles." (TASK-DOMAINS.md, line 39)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for a high-confidence decision.
