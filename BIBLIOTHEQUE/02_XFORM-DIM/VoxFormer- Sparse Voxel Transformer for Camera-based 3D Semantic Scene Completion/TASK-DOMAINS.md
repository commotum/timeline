# VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion (Not specified in the paper.)
Source: VoxFormer- Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scene completion (geometry occupancy) | RGB image(s) (current and previous frames) | 2D (x, y); 3D (x, y, z) or (x, y, t) (inferred) | Capped (inferred) | Dynamic | Constructed (inferred) | Binary occupancy voxel grid | 3D (x, y, z) or (x, y, t) | Fixed |
| Semantic segmentation (per-voxel semantics) | RGB image(s) (current and previous frames) | 2D (x, y); 3D (x, y, z) or (x, y, t) (inferred) | Capped (inferred) | Dynamic | Constructed (inferred) | Dense semantic voxel grid (M+1 classes including empty) | 3D (x, y, z) or (x, y, t) | Fixed |

## Summary
The paper covers camera-based semantic scene completion and reports two task intents: geometry-only scene completion and per-voxel semantic segmentation. Both tasks consume RGB image observations (single-image or temporal multi-image) and produce 3D voxel-grid outputs. The output interface is fixed to a predefined voxel volume, while input is capped to a bounded temporal window in the reported setups (inferred). Attention is dynamic via depth-based query selection and deformable attention, and state is constructed through learnable voxel queries, mask tokens, and refined voxel features (inferred).

## Evidence
### Task: Scene completion (geometry occupancy)
- "we use as input current and previous images denoted by  $\\mathbf{I}_t = \\{I_t, I_{t-1}, ...\\}$" (Section 3.1 Problem setup)
- "Our stage-1 determines which voxels to be queried based on depth" (Section 3.4 Stage-1: Class-Agnostic Ouery Proposal)
- "We employ intersection over union (IoU) to evaluate the scene completion quality, regardless of the allocated semantic labels." (Section 4.1 Evaluation metrics)
- "Such a group of geometryonly voxel grids is actually a binary occupancy map" (Section 4.1 Evaluation metrics)
- Inference: Input includes temporal indexing and capped dynamics because the framework "supports the input of single or multiple images" and is instantiated with fixed windows ("current" or "current and the previous 4 images"). State is Constructed because voxel queries and mask tokens are learnable parameters used to build refined 3D voxel features (Section 3.3, Figure 2 caption, Section 4.1 Implementation details).

### Task: Semantic segmentation (per-voxel semantics)
- "VoxFormer consists of class-agnostic query proposal (stage-1) and class-specific semantic segmentation (stage-2)" (Section 1 Introduction)
- "Output dense semantic map  $\\mathbf{Y}_t \\in \\mathbb{R}^{H \\times W \\times Z \\times (M+1)}$" (Section 3.2 Overall Architecture)
- "the full set of voxels will be processed by self-attention to complete the scene representations for per-voxel semantic segmentation." (Section 1 Introduction)
- "We use the mean IoU (mIoU) of 19 semantic classes to assess the performance of semantic segmentation." (Section 4.1 Evaluation metrics)
- Inference: The same input-dimension/dynamics inference applies here because stage-2 consumes the same image stream and temporal variants. State is Constructed (inferred) because stage-2 combines updated query proposals with mask tokens and refines them via deformable self-attention into full-scene voxel features (Section 3.5).
