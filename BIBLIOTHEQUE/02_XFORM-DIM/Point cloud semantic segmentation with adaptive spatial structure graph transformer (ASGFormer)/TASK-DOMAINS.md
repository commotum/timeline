# Point cloud semantic segmentation with adaptive spatial structure graph transformer (2024)
Source: Point cloud semantic segmentation with adaptive spatial structure graph transformer (ASGFormer).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point cloud semantic segmentation | 3D point cloud (points) | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | per-point semantic labels | 3D (x, y, z) | Not specified in the paper. |

## Summary
The paper focuses on a single task: semantic segmentation of 3D point clouds, producing per-point semantic labels. Inputs are explicitly defined as 3D point sets, and outputs are generated for each point in the original cloud. The paper does not state fixed size constraints on input/output dynamics. The neighborhood selection is fixed-radius, so attention is treated as static (inferred), and the model builds intermediate graph/feature representations, indicating constructed state (inferred).

## Evidence
### Task: point cloud semantic segmentation
- "With the rapid development of LiDAR and artificial intelligence technologies, 3D point cloud semantic segmentation has become a highlight research topic." (Abstract)
- "we propose a Graph Transformer point cloud semantic segmentation network (ASGFormer) tailored for structurally adherent objects." (Abstract)
- "Given an input set of points  $P = \{P_n | n = 1, 2, \dots, N; P_n \in \mathbb{R}^3\}$ , where N denotes the number of points." (Section 3.2)
- "In the final stage of the decoder, feature vector is computed for each point, and then MLP is employed to generate final segmentation results with  $N_{cls}$  dimension." (Section 3.1)
- Inference: Marked Attention Dynamic as Static because the neighbor set is fixed by sampling: "we employ the fix-radius farthest point sampling strategy to select  $N(i) = \{j; (j, i) \in E\} \cup \{i\}$  neighbor points for each vertex i" (Section 3.2).
- Inference: Marked State Dynamic as Constructed because the model aggregates and creates new point features: "the AGT block aggregates neighbor structural features and utilizes graph attention to generate new features for all points." (Section 3.2)
