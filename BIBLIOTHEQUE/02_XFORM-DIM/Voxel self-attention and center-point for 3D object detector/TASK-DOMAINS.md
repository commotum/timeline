# Voxel self-attention and center-point for 3D object detector (2024)
Source: Voxel self-attention and center-point for 3D object detector.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | LiDAR point clouds | 3D (x, y, z) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | 3D object detections (classes, center points, dimensions, orientation) | 3D (x, y, z); 0D | Capped (inferred) |

## Summary
The paper covers a single task: LiDAR-based 3D object detection for autonomous driving, evaluated on KITTI, Waymo Open Dataset, nuScenes, and campus driving scenes. Inputs are 3D point clouds, while outputs are object detections with class labels plus 3D center/box/orientation attributes. The paper explicitly caps output capacity and describes thresholded/voxelized preprocessing, supporting capped interface dynamics. Based on query-dependent voxel selection in self-attention and multi-stage learned feature construction, attention is dynamic (inferred) and state is constructed (inferred).

## Evidence
### Task: 3D object detection
- "To address these, an anchor-free 3D LiDAR object detector in VSAC with a larger receptive field is designed." (Section INTRODUCTION)
- "Unlike 2D images obtained from cameras, 3D point cloud data of LiDAR provides precise depth and spatial structure information" (Section INTRODUCTION)
- "a centerpoint detection head module for predicted classes and regression location information" (Section Methodology)
- "It can directly detect the center-point position and 3D dimensions of the object." (Section Center-point detection head module)
- "In the center-point detection head, the feature map scale factor is set at 1/4, with a maximum detection capacity of 100 objects" (Section Implementation details)
- Inference: `In Dynamics = Capped` is inferred from bounded voxelization/thresholding ("if the number of points in a voxel exceeds a preset threshold") and fixed coordinate ranges/voxel sizes in implementation details. `Attention Dynamic = Dynamic` is inferred from query-dependent selection ("for each  v_i , corresponding  v_k ∈ Ω(i) can be obtained" and "obtain the voxels participating in the attention mechanism"). `State Dynamic = Constructed` is inferred from learned intermediate abstractions ("average features of all points within a voxel are taken to represent the features of that voxel" and "PST-FPN processes feature learning and outputs a feature map with 256 channels"). `Out Dynamics = Capped` is inferred from "maximum detection capacity of 100 objects."
