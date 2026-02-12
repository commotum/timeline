# LeaF: Learning Frames for 4D Point Cloud Sequence Understanding (Year not specified)
Source: LeaF- Learning Frames for 4D Point Cloud Sequence Understanding.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The available auxiliary analysis states LeaF uses a transformer-style self-attention fusion as part of its main learning pipeline.
- The paper text and auxiliary analysis indicate transformer components/backbones (e.g., region frame-guided transformer, PPTr/P4Transformer) are used for the main reported results.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus the three available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "We formulate the fusion process as a self-attention operation" (TASK-DOMAINS.md, Evidence section citing Section 3.3)
- "region frame-guided transformer" (LeaF- Learning Frames for 4D Point Cloud Sequence Understanding.md, Figure 2 description)
- "As in action segmentation, we use PPTr as the base network." (TASK_MODEL_RATIO.md, item 2 quote from Section 4.2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md supports a transformer-central/hybrid model decision.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; extending-dimensions analysis markdown was unavailable (`MISSING`).
