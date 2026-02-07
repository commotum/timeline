# Mask4Former: Mask Transformer for 4D Panoptic Segmentation (Not specified in the paper.)
Source: Mask4Former- Mask Transformer for 4D Panoptic Segmentation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4D panoptic segmentation (LiDAR point clouds) | sequence of LiDAR scans / spatio-temporal point cloud | 4D (x, y, z, t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | semantic class label and instance ID per point (spatio-temporal instance masks) | 4D (x, y, z, t) | Not specified in the paper. |
| 3D panoptic segmentation | single LiDAR scan (3D point cloud) | 3D (x, y, z) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | semantic class label per point with instance distinctions | 3D (x, y, z) | Not specified in the paper. |
| 4D semantic segmentation | multiple LiDAR scans (spatio-temporal point cloud) | 4D (x, y, z, t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | semantic class labels per point (moving vs stationary classes) | 4D (x, y, z, t) | Not specified in the paper. |

## Summary
The paper addresses LiDAR point cloud segmentation across time, focusing on 4D panoptic segmentation and extending to 3D panoptic and 4D semantic segmentation in the supplementary material. The tasks operate over 3D or 4D (x, y, z, t) address spaces with point-wise labels and instance tracking where applicable, while input/output dynamics are not explicitly specified. The described architecture uses masked cross-attention and spatio-temporal instance queries, implying Dynamic attention and Constructed state (inferred).

## Evidence
### Task: 4D panoptic segmentation (LiDAR point clouds)
- "we propose Mask4Former for the challenging task of 4D panoptic segmentation of LiDAR point clouds." (Abstract)
- "given a sequence of LiDAR scans, the goal is to predict the semantic class of each point while consistently tracking object instances." (I. INTRODUCTION)
- "assign a single semantic class label and instance ID to every point within the spatio-temporal point cloud" (Extracting 4D panoptic segmentations)
- "ST queries attend only to the foreground voxels predicted by the previous mask module." (III. METHOD, Query Refinement Module)
- "spatio-temporal (ST) queries that encode geometric and semantic attributes of all instances in a sequence." (III. METHOD, Overview)
- Inference: Attention is Dynamic and State is Constructed based on masked cross-attention to predicted foreground voxels and the use of spatio-temporal instance queries (III. METHOD, Overview; Query Refinement Module).

### Task: 3D panoptic segmentation
- "Specifically, we use Mask4Former for both 3D panoptic segmentation and 4D semantic segmentation tasks." (Supplementary Material)
- "3D panoptic segmentation is the task of assigning a semantic class label for each point in a 3D scene" (Supplementary Material)
- "while distinguishing different instances of the same class." (Supplementary Material)
- "3D panoptic segmentation processes each LiDAR scan independently." (Supplementary Material)
- "ST queries attend only to the foreground voxels predicted by the previous mask module." (III. METHOD, Query Refinement Module)
- "spatio-temporal (ST) queries that encode geometric and semantic attributes of all instances in a sequence." (III. METHOD, Overview)
- Inference: Attention is Dynamic and State is Constructed because Mask4Former uses masked cross-attention and spatio-temporal instance queries in its described architecture (III. METHOD, Overview; Query Refinement Module).

### Task: 4D semantic segmentation
- "Specifically, we use Mask4Former for both 3D panoptic segmentation and 4D semantic segmentation tasks." (Supplementary Material)
- "4D semantic segmentation is a semantic segmentation task where moving and stationary objects of the same category are treated as different semantic classes." (Supplementary Material)
- "the model needs to process multiple LiDAR scans together." (Supplementary Material)
- "ST queries attend only to the foreground voxels predicted by the previous mask module." (III. METHOD, Query Refinement Module)
- "spatio-temporal (ST) queries that encode geometric and semantic attributes of all instances in a sequence." (III. METHOD, Overview)
- Inference: Attention is Dynamic and State is Constructed because Mask4Former uses masked cross-attention and spatio-temporal instance queries in its described architecture (III. METHOD, Overview; Query Refinement Module).
