# Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation (Year not specified)
Source: Spherical Mask- Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract positions Transformer-based methods as competing prior work and presents Spherical Mask itself as a coarse-to-fine spherical-representation approach.
- Auxiliary analysis indicates a voting + dynamic-convolution pipeline rather than Transformer self-attention blocks; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "Coarse-to-fine 3D instance segmentation methods show weak performances compared to recent Grouping-based, Kernel-based and Transformer-based methods. ... In this work, we introduce **Spherical Mask**, a novel coarse-to-fine approach based on spherical representation..." (Abstract, `Spherical Mask- Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation.md`:9-11)
- "The model uses query-dependent vote generation and dynamic convolution over learned features, supporting dynamic attention and constructed state." (`TASK-DOMAINS.md`:10)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO decision from abstract framing and auxiliary model-family cues.
Pass 2 (targeted source scan): skipped - Not needed after Pass 1; extending-dimensions analysis was unavailable (`MISSING`).
