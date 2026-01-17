# Swin Transformer V2: Scaling Up Capacity and Resolution (Not specified in the paper.)
Source: Swin Transformer V2- Scaling Up Capacity and Resolution.md

## Core reasons
- The paper presents multiple architectural and training-method changes (normalization, attention, position bias, self-supervised pretraining) to scale vision Transformers rather than a single positional-encoding-only contribution.
- The focus is on scaling capacity/resolution and stabilizing training for vision tasks, which fits broader ML modeling and training methodology instead of datasets, benchmarks, or computation mechanisms.

## Evidence extracts
- "Three main techniques are proposed: 1) a residual-post-norm method combined with cosine attention to improve training stability; 2) A log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) A self-supervised pretraining method, SimMIM, to reduce the needs of vast labeled images." (Abstract)
- "To better scale up model capacity and window resolution, several adaptions are made on the original Swin Transformer architecture (V1): 1) A res-post-norm to replace the previous pre-norm configuration; 2) A scaled cosine attention to replace the original dot product attention; 3) A log-spaced continuous relative position bias approach to replace the previous parameterized approach." (Figure 1)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
