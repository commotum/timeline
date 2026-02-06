# 4D-Former: Multimodal 4D Panoptic Segmentation (Not specified in the paper)
Source: 4D-Former- Multimodal 4D Panoptic Segmentation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4D panoptic segmentation | LiDAR point-cloud sequence; RGB camera images | 4D (x, y, z, t); 2D (x, y) | Open | Static (inferred) | Constructed (inferred) | per-point semantic class labels; object track masks/track IDs | 4D (x, y, z, t) | Open (inferred) |

## Summary
The paper defines a single task: 4D panoptic segmentation that assigns semantic labels and temporally consistent instance IDs to LiDAR point-cloud sequences, using both LiDAR and RGB camera images. Inputs span 4D spatiotemporal point clouds and 2D images, with outputs as per-point semantic and track masks over time, and the system is designed to handle arbitrarily long streams via sliding windows. Attention is treated as static and the model state as constructed based on its query-based fusion over fixed features and the explicit track memory bank.

## Evidence
### Task: 4D panoptic segmentation
- "4D panoptic segmentation is a challenging but practically useful task" (Abstract)
- "requires every point in a LiDAR point-cloud sequence to be assigned a semantic class label" (Abstract)
- "and individual objects to be segmented and tracked over time." (Abstract)
- "takes as input the current LiDAR scan at time t, the past scan at t-1, and the camera images at time t." (Section 3 Multimodal 4D Panoptic Segmentation)
- "predicts per-point semantic and object track masks" (Section 3.2 Transformer-based Panoptic Decoder)
- "handle sequences of arbitrary length as well as continuous streams of data" (Section 3 Multimodal 4D Panoptic Segmentation)
- "we maintain a memory bank containing object tracks." (Section 3.3 Tracklet Association Module)
- "cross-attending to the voxel features" (Section 3.2 Transformer-based Panoptic Decoder)
- "cross-attending to the set of image features" (Section 3.2 Transformer-based Panoptic Decoder)
- Inference: Attention Dynamic marked Static (inferred) because the model cross-attends over fixed voxel/image features without runtime input selection; State Dynamic marked Constructed (inferred) because it maintains a track memory bank; Out Dynamics marked Open (inferred) because outputs follow arbitrarily long input streams. (Sections 3, 3.2, 3.3)
