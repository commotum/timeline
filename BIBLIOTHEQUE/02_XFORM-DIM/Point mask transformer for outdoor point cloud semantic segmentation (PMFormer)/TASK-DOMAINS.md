# Point mask transformer for outdoor point cloud semantic segmentation (2025)
Source: Point mask transformer for outdoor point cloud semantic segmentation (PMFormer).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic segmentation | 3D LiDAR point cloud | 3D (x, y, z) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | binary point masks + class labels | 3D (x, y, z) | Open (inferred) |

## Summary
The paper addresses a single task: semantic segmentation of outdoor 3D LiDAR point clouds, producing class-labeled binary masks over the input points. Inputs and outputs are spatial 3D (x, y, z) point-cloud domains, and the paper explicitly avoids constraining the number of points, so the interface is treated as Open by inference. The attention mechanism is described as foreground-weighted cross-attention and the model uses learned query embeddings, which we interpret as Dynamic attention and Constructed state (both inferred from the architectural description).

## Evidence
### Task: semantic segmentation
- "Specifically, we performed semantic segmentation of 3D LiDAR point clouds by directly predicting a set of binary masks and their corresponding semantic classes." (Section 2.3 Transformer in point cloud)
- "The objective of point-cloud semantic segmentation is to partition an entire point-cloud scene into distinct regions based on their respective categories." (Section 1 Introduction)
- "It uses a point cloud  $P \in \mathbb{R}^{N \times 4}$  as the input, where N is the number of points in the point cloud." (Section 3.2.1 Sparse point-voxel convolution network)
- "The goal of mask classification is to predict a set of binary masks  $m_i \in [0,1]^L$ , each associated with a single-class prediction  $p_i \in C^{K+1}$ ." (Section 3.1 3D mask classification)
- Inference: In/Out Dynamics labeled Open because "we refrain from constraining the number of points" and the number of per-point embeddings differs by frame (Section 3.3.2 3D position encoding). Attention Dynamic labeled Dynamic because "each query attends only to the foreground region" via a weight map (Section 3.3.3 Attention weights). State Dynamic labeled Constructed because the model "initialize[s] a set of queries" that are processed by the transformer to produce per-segment embeddings beyond the raw input (Introduction).
