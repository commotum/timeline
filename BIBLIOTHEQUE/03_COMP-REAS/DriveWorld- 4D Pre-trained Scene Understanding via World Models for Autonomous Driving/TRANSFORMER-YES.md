# DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving (Year not specified)
Source: DriveWorld- 4D Pre-trained Scene Understanding via World Models for Autonomous Driving.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: medium
Basis: source-targeted-scan

## Why
- The core DriveWorld method description includes Transformer-based components in the main world-model pipeline, not only in related work or baselines.
- The temporal module explicitly uses an attention mechanism (cross-attention) as part of its state update path, indicating attention is materially used in the central model.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), so the decision relies on the abstract, available auxiliary files, and targeted method-section cues.

## Evidence
- "As shown in Fig. 2, the designed world model consists of an Image Encoder, a 2D to 3D View Transform (e.g., Transformers [77], LSS [62] techniques), a Memory State-Space Model ..." (DriveWorld- 4D Pre-trained Scene Understanding via World Models for Autonomous Driving.md, Section 3, line 65)
- "The refined deterministic history  $\tilde{h}_t$  is obtained via the cross-attention mechanism with the dynamics memory bank." (DriveWorld- 4D Pre-trained Scene Understanding via World Models for Autonomous Driving.md, Section 3.1, line 94)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read the abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; Extending-dimensions analysis markdown was unavailable (`MISSING`), and Pass 1 evidence alone did not explicitly settle central Transformer usage.
Pass 2 (targeted source scan): performed - Method scan found Transformer-based view transform and cross-attention in the main DriveWorld architecture, sufficient to finalize.
