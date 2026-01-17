# Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion (Not specified in the paper.)
Source: Sparse4D- Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion.md

## Core reasons
- The paper proposes a multi-view 3D detection model that explicitly fuses spatial-temporal (4D) features for 3D perception, aligning with transformer adaptations for higher-dimensional domains.
- The method uses an encoder-decoder with self-attention and a deformable 4D aggregation module to handle multi-view, multi-scale, and multi-timestamp features, indicating attention-based modeling beyond 1D sequences.

## Evidence extracts
- "we introduce a novel method, named Sparse4D, which does the iterative refinement of anchor boxes via sparsely sampling and fusing spatial-temporal features." (Abstract)
- "In each refinement module, we first adopt self-attention to realize the interaction between instances, with the embedding of anchor parameters added before and after. Then, we conduct deformable 4D aggregation (Sec. 3.2) to fuse multi-view, multi-scale, multi-timestamp and multi-keypoint features." (Section 3.1. Overall Framework)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
