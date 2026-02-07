# Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation (Not specified in the paper)
Source: Relation3D- Enhancing Relation Modeling for Point Cloud Instance Segmentation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D point cloud instance segmentation | 3D point cloud scenes with per-point position/color/normal features | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Instance binary foreground masks with semantic category labels | 3D (x, y, z); 0D | Capped (inferred) |

## Summary
Relation3D covers one task: 3D point cloud instance segmentation across ScanNetV2, ScanNet++, ScanNet200, and S3DIS. The task operates on 3D spatial point-cloud input and outputs 3D instance masks plus per-instance semantic labels. Input-size dynamics are not explicitly specified in the OCR text, while output cardinality is capped by the configured query count K (inferred from the decoder/query design). The architecture uses static attention over provided query/scene features and constructed internal state via superpoint/query feature refinement (both inferred).

## Evidence
### Task: 3D point cloud instance segmentation
- "3D instance segmentation aims to predict a set of object instances in a scene, representing them as binary foreground masks with corresponding semantic labels." (Section Abstract)
- "Point cloud instance segmentation aims to identify and segment multiple instances of specific object categories in 3D space." (Section 1. Introduction)
- "The goal of 3D instance segmentation is to determine the categories and binary masks of all foreground objects in the scene." (Section 3.1. Overview)
- "Assuming that the input point cloud has N points, each point contains position (x, y, z), color (r, q, b)and normal  $(n_x, n_y, n_z)$  information." (Section 3.1. Overview)
- "Subsequently, we initialize several instance queries  $Q \in \mathbb{R}^{K \times C}$  and input Q and  $F_{\text{super}}$  into the transformer decoder." (Section 3.1. Overview)
- "For hyperparameters, we tune K, r as 400, 3 respectively. Since ScanNet++ and ScanNet200 have more categories and instances, we set K as 500." (Section 4.1. Experimental Setup)
- Inference: `Attention Dynamic = Static` is inferred because the method applies decoder self/cross-attention over provided query and superpoint features rather than runtime retrieval/selection beyond the given inputs ("the superpoint refinement module employs a cross-attention mechanism..." and "we propose a relation-aware self-attention (RSA)." in Sections 3.4 and 3.5). `State Dynamic = Constructed` is inferred because the model explicitly builds and iteratively updates superpoint/query representations ("forms a dual-path architecture, enabling direct communication between query and superpoint features." in Section 3.4). `Out Dynamics = Capped` is inferred from the explicit query-count cap K used to generate instance predictions (Sections 3.1 and 4.1). `In Dynamics` remains not specified because no explicit maximum input-point count/interface cap for N is stated.
