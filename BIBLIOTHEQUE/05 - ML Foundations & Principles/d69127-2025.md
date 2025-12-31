# Zero-Shot 4D Lidar Panoptic Segmentation (Not specified in the paper.)
Source: d69127-2025.pdf

## Core reasons
- Proposes SAL-4D with a pseudo-labeling engine that distills video object segmentation and CLIP features to enable zero-shot 4D Lidar panoptic segmentation and tracking.
- Focuses on a modeling/training approach for spatio-temporal Lidar segmentation and recognition rather than introducing a dataset/benchmark, positional encoding change, or computation mechanism.

## Evidence extracts
- "To train our SAL-4D for Zero-Shot 4D Lidar Panoptic
Segmentation, we present a pseudo-labeling engine that is
built on the insight that we can reliably prompt state-of-the-
art VOS models over short temporal horizons in videos and
generate their corresponding sequence-level CLIP features
to facilitate zero-shot recognition. To account for inherently" (Introduction)
- "(3D)pointcloudsinisolation. In contrast, our data-driven approach (right) operates directly on sequences of point clouds, jointly perform-
ing object segmentation, tracking, and zero-shot recognition based on text prompts specified at test time. Our method localizes and tracks
any object and provides a temporally coherent semantic interpretation of dynamic scenes. We can correctly segment canonical objects," (Figure 1 caption)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
