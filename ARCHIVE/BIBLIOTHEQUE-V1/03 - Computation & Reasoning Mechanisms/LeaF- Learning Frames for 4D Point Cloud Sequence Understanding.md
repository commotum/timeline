# LeaF: Learning Frames for 4D Point Cloud Sequence Understanding (2023)
Source: 91102e-2023.pdf

## Core reasons
- LeaF learns region-wise coordinate frames so that spatial and motion features can be factorized, enabling more effective computation on 4D point cloud sequences across timestamps.
- A frame-guided two-tower pipeline fuses motion-invariant region frame features with camera-frame motion cues via attention, changing how 4D features are computed and thus the reasoning process.

## Evidence extracts
- "We focus on learning descriptive geometry and motion features from 4D point cloud sequences in this work. Existing works usually develop generic 4D learning tools without leveraging the prior that a 4D sequence comes from a single 3D scene with local dynamics. Based on this observation, we propose to learn region-wise coordinate frames that transform together with the underlying geometry. With such frames, we can factorize geometry and motion to facilitate a feature-space geometric reconstruction for more effective 4D learning." (p. 604)
- "Frame-guided feature learning. With the frame-aware 4D operation we can extract hierarchical region frame features, which are inherent representations of the underlying geometry but lacks the motion information... we propose to use the region frame feature to guide the camera frame feature learning through an attention map which aligns the local region across different timestamps into a canonical space and allows easier temporal associations for better motion understanding..." (p. 609)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
