# Transformer based 3D tooth segmentation via point cloud region partition (2024)
Source: Transformer based 3D tooth segmentation via point cloud region partition (PointRegion).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D tooth semantic segmentation | 3D dental point cloud sampled from mesh data | 3D (x, y, z) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Point-wise tooth/gingiva class logits and labels | 3D (x, y, z) | Capped (inferred) |

## Summary
The paper covers one task: 3D tooth semantic segmentation on dental point clouds converted from mesh models. The task maps 3D point-cloud input to point-wise tooth/gingiva labels, so both input and output are grounded in 3D (x, y, z). Based on the described pipeline, dynamics are Capped (inferred) because training and testing use fixed-size 10240-point sub-samples and fixed neighborhood sizes. Attention Dynamic and State Dynamic are inferred as Dynamic and Constructed because the model selects K-nearest neighbor regions per point and builds region embeddings plus a learned point-to-region probability matrix.

## Evidence
### Task: 3D tooth semantic segmentation
- "Automatic and accurate tooth segmentation on 3D dental point clouds plays a pivotal role in computeraided dentistry." (Abstract)
- "In this paper, we design a PointRegion model for 3D tooth segmentation." (Section Overview)
- "Given an input mesh dental model, we sample it to obtain the point cloud consisting of N points, each of which has d-Dimensional attributes." (Section Mesh2Point)
- "Finally, with the help of the region logits  Y^p  introduced in Section RegionEncoder module, we calculate the class logits for each point  p_i$ , denoted as  Y^p = \{y_i^p : y_i^p \in \mathbb{R}^C, i = 1, \dots, N\}$ , through weighted summation as Eq. (10):" (Section Point level segmentation based on point and region association)
- Inference: In Dynamics and Out Dynamics are Capped (inferred) because "We split all points into three sub-samples including 10240 points using FPS" and "we also use multiple FPS to get multiple sub-samples with the size of 10240" (Section Implementation details). Attention Dynamic is Dynamic (inferred) because "Via K-nearest neighbor (KNN) algorithm, we can find K neighboring regions of point p_i" (Section Searching for K-nearest neighbor regions (SKNR)). State Dynamic is Constructed (inferred) because the method is built by "interpreting the point cloud as a tessellation of learnable regions," learns "region embeddings," and forms a learned "probability matrix S" for point-to-region association (Sections Overview; RegionPartition module; Learning point-to-region probability (LP)).
