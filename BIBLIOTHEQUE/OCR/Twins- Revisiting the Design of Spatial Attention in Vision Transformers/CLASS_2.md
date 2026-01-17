# Twins: Revisiting the Design of Spatial Attention in Vision Transformers (Not specified in the paper.)
Source: Twins- Revisiting the Design of Spatial Attention in Vision Transformers.md

## Core reasons
- Proposes new vision transformer architectures for visual tasks like classification, detection, and segmentation, indicating a transformer adaptation for 2D image domains.
- Redesigns spatial attention with a new local/global mechanism to address the spatial self-attention challenges of high-resolution images.

## Evidence extracts
- "As a result, we propose two vision transformer architectures, namely, Twins-PCPVT and Twins-SVT. Our proposed architectures are highly efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks including image-level classification as well as dense detection and segmentation." (Abstract)
- "Here, we propose the spatially separable self-attention (SSSA) to alleviate this challenge. SSSA is composed of locally-grouped self-attention (LSA) and global sub-sampled attention (GSA)." (Section 3.2 Twins-SVT)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
