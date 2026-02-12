# Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance (Year not specified)
Source: Open3DIS- Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The method pipeline explicitly uses Transformer-based components in core processing: Grounding-DINO + SAM for 2D mask generation and CLIP ViT-L/14 for open-vocabulary feature matching.
- These components are part of the main Open3DIS pipeline used for reported results, not only mentioned as related-work baselines.
- The Extending-dimensions analysis file was unavailable (`MISSING`), so the decision used the abstract, available auxiliary files, and targeted architecture cues from the source.

## Evidence
- "Per-frame superpoint merging. For all input frames, we utilize a pretrained 2D instance segmenter, employing Grounding-DINO [32] and SAM [26]." (Open3DIS- Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance.md, Section 3.1, line 87)
- "For CLIP, we use the ViT-L/14 [39]." (Open3DIS- Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance.md, Section 4.1 Implementation Details, line 145)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - abstract and auxiliary files confirmed the pipeline and CLIP-based open-vocabulary matching; Extending-dimensions analysis was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - explicit Transformer-family cues (Grounding-DINO, SAM, ViT-L/14) were found in method/implementation details, making the decision high-confidence.
