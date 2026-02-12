# Per-Pixel Classification is Not All You Need for Semantic Segmentation (MaskFormer) (Year not specified)
Source: Per-Pixel Classification is Not All You Need for Semantic Segmentation (MaskFormer).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s method description makes a Transformer decoder a core module for generating segmentation predictions, so self-attention is material to the central model.
- Auxiliary analysis files also identify a transformer decoder attending to image features as part of the main architecture.
- The Extending-dimensions analysis file was unavailable (`MISSING`), so the decision is based on the abstract and available auxiliary files.

## Evidence
- "Using the set prediction mechanism proposed in DETR [3], MaskFormer employs a Transformer decoder [37] to compute a set of pairs," (Per-Pixel Classification is Not All You Need for Semantic Segmentation (MaskFormer).md, Section 1 Introduction)
- "A transformer decoder attends to image features and produces N per-segment embeddings." (TASK-DOMAINS.md, Evidence section, Figure 2 caption quote)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient, high-confidence Transformer evidence found in the abstract/main OCR and auxiliary analysis; `MISSING` extending-dimensions file skipped as unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 was already decisive.
