# Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation (Not specified in the paper)
Source: Spherical Mask- Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D point cloud instance segmentation | Point clouds with corresponding color information | 3D (x, y, z) or (x, y, t) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Local binary instance masks with per-instance class labels and confidence scores | 3D (x, y, z) or (x, y, t); 0D | Capped (inferred) |

## Summary
The paper covers one primary task: 3D point cloud instance segmentation from colored point clouds. Inputs and mask outputs are defined over a 3D spatial domain, with additional 0D per-instance outputs for class and confidence. The interface is variable-size but bounded through fixed proposal counts and configured sampling, so dynamics are best supported as capped. The model uses query-dependent vote generation and dynamic convolution over learned features, supporting dynamic attention and constructed state.

## Evidence
### Task: 3D point cloud instance segmentation
- "Similar to 2D instance segmentation, the goal of the task is to identify each object along with its class label." (Section 1. Introduction)
- "Given an input point cloud  $p_1 \in \mathbb{R}^{N_p \times 3}$  in 3-dimensional cartesian coordinates and the corresponding color information  $p_{\text{rgb}} \in \mathbb{R}^{N_p \times 3}$ , we aim to design a system that segments the point cloud into local binary masks of instances  $\{o^{(i)} \in \mathbb{R}^{N_p \times 1}\}_{i=1}^{N_o}$  using a coarse to fine approach." (Section 3.1. Overview)
- "In Mask Assembly, *K* local binary masks are generated, where each mask is a proposal for a single instance. 3D NMS is applied to acquire the final instance masks using local binary masks, classifications, and confidence scores." (Figure 2 caption, Section 3)
- Inference: In Dynamics and Out Dynamics are labeled Capped because the paper describes variable cardinalities ($N_p$, $N_o$) while also using fixed proposal/vote interfaces: "*K* local binary masks are generated" and "The number seeds and votes are set to 1024 and 256, respectively." (Figure 2 caption; Section 4.3. Implementation Detail). Attention Dynamic is labeled Dynamic because runtime queries are used: "producing K votes with query points" and dynamic convolution predicts offsets "using the vote features  $F_2$ ... as queries against the point features  $F_1$" (Section 3.2; Section 3.3.2). State Dynamic is labeled Constructed because the system constructs intermediate representations before output: "encode the given point cloud into deep features  $F_1$" and "producing K votes with query points  $p_2$  and features  $F_2$" (Section 3.2).
