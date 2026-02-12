# Segment Anything (2023)
Source: Segment Anything.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The auxiliary analysis states that SAM’s core image encoder is a pre-trained ViT-H, which is a Transformer-family architecture used by the main model.
- The auxiliary analysis also identifies prompt/image cross-attention in SAM’s architecture, indicating attention mechanisms are central to SAM rather than peripheral.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation." (Abstract, Segment Anything.md)
- "**Implementation.** Unless otherwise specified: (1) SAM uses an MAE [47] pre-trained ViT-H [33] image encoder and (2) SAM was trained on SA-1B..." (§7 quote in TASK_MODEL_RATIO.md)
- "Attention and state labels are inferred from the architecture description: prompt/image cross-attention over provided inputs and reusable image embeddings across prompts." (Summary, TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-YES from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already sufficient for a high-confidence decision.
