# Voxel Transformer for 3D Object Detection (Not specified in the paper)
Source: Voxel Transformer for 3D Object Detection (VoTr).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | Point clouds rasterized into sparse voxels | 3D (x, y, z) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | 3D bounding boxes (proposals/detections) | 3D (x, y, z) | Capped (inferred) |

## Summary
The paper covers one core task: 3D object detection from point clouds, implemented through sparse voxel representations. Both the input and output operate in 3D spatial domains that are explicitly described as voxelized 3D scenes and 3D proposals/boxes. The input/output size behavior is best supported as capped because the method rasterizes scenes into a finite dense voxel grid and generates proposals on BEV features. Attention and state behavior are inferred as dynamic and constructed, respectively, from query-dependent voxel selection and multi-stage learned feature construction.

## Evidence
### Task: 3D object detection
- "We present Voxel Transformer (VoTr), a novel and effective voxel-based Transformer backbone for 3D object detection from point clouds." (Abstract)
- "Voxel-based detectors transform irregular point clouds into regular voxel-grids and show superior performance in this task." (Section 1. Introduction)
- "Voxel features extracted by our proposed VoTr are then projected to a BEV feature map to generate 3D proposals" (Section 3.1. Overall Architecture)
- "We define a dense voxel-grid, which has  $N_{dense}$  voxels in total, to rasterize the whole 3D scene. In practice we only maintain those non-empty voxels with a  $N_{sparse} \times 3$  integer indices array V and  $N_{sparse} \times d$  corresponding feature array  $\mathcal{F}$  for efficient computation, where  $N_{sparse}$  is the number of non-empty voxels and  $N_{sparse} \ll N_{dense}$ ." (Section 3.2. Voxel Transformer Module)
- "Figure 5 shows that a querying voxel can dynamically select the features of attending voxels in a very large context range" (Section 4.4. Ablation Studies)
- Inference: In Dynamics is labeled Capped from the finite dense voxel grid with variable non-empty occupancy (Section 3.2). Attention Dynamic is labeled Dynamic from query-dependent attending-voxel selection and explicit dynamic selection language (Sections 3.1 and 4.4). State Dynamic is labeled Constructed because the method builds and reuses intermediate voxel/BEV/proposal representations for detection and refinement (Sections 3.1 and 4.1). Out Dynamics is labeled Capped because detections are produced as 3D proposals on finite BEV features in anchor-based detector frameworks (Sections 3.1 and 4.1).
