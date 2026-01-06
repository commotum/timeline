# CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification (Not specified in the paper.)
Source: CrossViT- Cross-Attention Multi-Scale Vision Transformer for Image Classification.md

## Core reasons
- The paper proposes a vision-transformer architecture for image classification that operates on image patches at multiple scales using a dual-branch design.
- The central contribution is a multi-scale transformer adaptation for images that fuses different patch-size representations via cross-attention.

## Evidence extracts
- "we study how to learn multi-scale feature representations in transformer models for image classification." (Section 1. Introduction)
- "Our model is primarily composed of K multiscale transformer encoders where each encoder consists of two branches: (1) **L-Branch**: a large (primary) branch that utilizes coarse-grained patch size  $(P_l)$  with more transformer encoders and wider embedding dimensions, (2) **S-Branch**: a small (complementary) branch that operates at fine-grained patch size  $(P_s)$  with fewer encoders and smaller embedding dimensions." (Section 3.2. Proposed Multi-Scale Vision Transformer)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
