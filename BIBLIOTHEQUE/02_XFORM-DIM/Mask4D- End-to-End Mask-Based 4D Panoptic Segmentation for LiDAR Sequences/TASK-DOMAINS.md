# Mask4D: End-to-End Mask-Based 4D Panoptic Segmentation for LiDAR Sequences (2023)
Source: Mask4D- End-to-End Mask-Based 4D Panoptic Segmentation for LiDAR Sequences.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4D panoptic segmentation | 3D LiDAR scan sequences | 4D (x, y, z, t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Semantic classes and temporally consistent instance IDs per point | 4D (x, y, z, t) | Not specified in the paper. |
| 3D panoptic segmentation | 3D LiDAR scans | 3D (x, y, z) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Semantic classes and instance IDs per point | 3D (x, y, z) | Not specified in the paper. |

## Summary
Mask4D covers 3D panoptic segmentation of individual LiDAR scans and 4D panoptic segmentation over LiDAR scan sequences with temporally consistent instance IDs. The inputs and outputs are point-level semantic/instance labels over 3D point clouds (3D) and their spatiotemporal sequences (4D), while explicit size dynamics are not specified. The paper describes mask attention and query reuse for tracking, which implies dynamic attention and constructed state across scans.

## Evidence
### Task: 4D panoptic segmentation
- "In this paper, we investigate the problem of 4D panoptic segmentation for 3D LiDAR scans" (Section I. Introduction)
- "4D panoptic segmentation further extends this information with temporarily consistent instance IDs" (Abstract)
- "directly outputs for each point a semantic class and instance IDs which are consistent over time." (Section III.B Mask4D for 4D Panoptic Segmentation)
- Inference: Attention Dynamic marked Dynamic (inferred) because "Mask attention [6] is a variation of cross-attention that only attends within the foreground region of a binary mask for each query i" (Section III.E Position-aware Mask Attention).
- Inference: State Dynamic marked Constructed (inferred) because "The tracking queries carry the identity of the instances, allowing us to keep consistent instance IDs over time." (Section III. Our Approach, Fig. 3 description)

### Task: 3D panoptic segmentation
- "Panoptic segmentation of 3D LiDAR scans allows us to semantically describe a vehicle's environment" (Abstract)
- "predicting semantic classes for each 3D point and to identify individual instances through different instance IDs." (Abstract)
- "Our approach uses the same network to perform 3D and 4D panoptic segmentation without relying on any post-processing step" (Section I. Introduction)
- Inference: Attention Dynamic marked Dynamic (inferred) because "Mask attention [6] is a variation of cross-attention that only attends within the foreground region of a binary mask for each query i" (Section III.E Position-aware Mask Attention).
- Inference: State Dynamic marked Constructed (inferred) because "The tracking queries carry the identity of the instances, allowing us to keep consistent instance IDs over time." (Section III. Our Approach, Fig. 3 description)
