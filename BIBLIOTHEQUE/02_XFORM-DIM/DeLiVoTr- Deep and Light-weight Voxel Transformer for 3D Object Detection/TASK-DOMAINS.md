# DeLiVoTr: Deep and light-weight voxel transformer for 3D object detection (2024)
Source: DeLiVoTr- Deep and Light-weight Voxel Transformer for 3D Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | LiDAR point cloud | 3D (x, y, z) | Capped (inferred) | Static (inferred) | Constructed (inferred) | 3D bounding boxes (object detections) | 3D (x, y, z) | Capped (inferred) |

## Summary
The paper addresses a single task: LiDAR-based 3D object detection in autonomous driving, taking LiDAR point clouds and producing 3D bounding box detections. Inputs are 3D spatial point sets and outputs are 3D spatial detections; input and output sizes are bounded by the fixed perception range/BEV grid, implying capped dynamics (inferred). Attention operates over predefined voxel regions and the model constructs latent voxel/region features before decoding (Static attention and Constructed state, inferred).

## Evidence
### Task: 3D object detection
- "3D object detection is one of the fundamental tasks in autonomous driving and robotics." (Section 1. Introduction)
- "The DeLiVoTr inputs LiDAR point cloud and predicts 3D bounding boxes in an autonomous driving scenario." (Section 3.1. Overview)
- Inference: In Dynamics = Capped (inferred) because "The point cloud is discretized into a sparse grid of shape  $(s_x, s_y, s_z)$ ." and dynamic voxelization yields a "dynamic number of voxels and points," implying variable but bounded input size (Section 3.2. Voxelization). Attention Dynamic = Static (inferred) because "we divide the voxels ( $\mathcal{V}$ ) into non-overlapping regions" and compute interactions within each region (Section 3.3.2. Voxel intra-region transformer). State Dynamic = Constructed (inferred) because they "employ dynamic voxel feature encoding (VFE) ... to transform the sparse voxels into a latent space representation" (Section 3.1. Overview). Out Dynamics = Capped (inferred) because voxel features are "rasterized to a BEV feature map" and the detection head "predicts class-specific heatmap" over that fixed grid (Section 3.4. Decoder).

---

## CSV Output (required)
Write a CSV file to "/home/jake/Developer/timeline/BIBLIOTHEQUE/02_XFORM-DIM/DeLiVoTr- Deep and Light-weight Voxel Transformer for 3D Object Detection/.TASK-DOMAINS.csv.tmp.398ae7da269a41a9a1b932440a7117a0" with the same rows and columns as the Task
Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
