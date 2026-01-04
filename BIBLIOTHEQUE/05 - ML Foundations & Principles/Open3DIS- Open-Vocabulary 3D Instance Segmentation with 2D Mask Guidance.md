# Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance (Not specified in the paper.)
Source: ab2e9f-2024.pdf

## Core reasons
- Presents Open3DIS and its 2D-guided 3D proposal plus pointwise feature extraction pipeline as a model/algorithm contribution for 3D instance segmentation.
- Uses point cloud CLIP features with text embeddings to generate instance masks, indicating a method for open-vocabulary 3D segmentation rather than positional encoding or transformer dimensional lifting.

## Evidence extracts
- "Open3DIS:Open-Vocabulary3DInstanceSegmentationwith2DMaskGuidance" (p. 1)
- "                   Figure 2. Overview of Open3DIS. A pre-trained class-agnostic 3D Instance Segmenter proposes initial 3D objects, while a 2D Instance
                   Segmenter generates masks for video frames. Our 2D-Guided-3D Instance Proposal Module (Sec. 3.1) combines superpoints and 2D
                   instance masks to enhance 3D proposals, integrating them with the initial 3D proposals. Finally, the Pointwise Feature Extraction module
                   (Sec. 3.3) correlates instance-aware point cloud CLIP features with text embeddings to generate the ultimate instance masks." (p. 3)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
