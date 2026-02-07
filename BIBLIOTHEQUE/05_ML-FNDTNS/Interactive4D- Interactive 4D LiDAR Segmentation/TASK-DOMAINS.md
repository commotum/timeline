# Interactive 4D LiDAR Segmentation (Not specified in the paper.)
Source: Interactive4D- Interactive 4D LiDAR Segmentation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Interactive LiDAR segmentation (single scan) | LiDAR point cloud (single scan) with user clicks | 3D (x, y, z) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Per-point object masks / instance IDs | 3D (x, y, z) | Not specified in the paper. |
| Interactive 4D LiDAR segmentation and tracking | Spatio-temporal point cloud (superimposed consecutive scans) with user clicks | 4D (x, y, z, t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Per-point object masks with consistent instance IDs over time | 4D (x, y, z, t) | Not specified in the paper. |

## Summary
The paper covers interactive LiDAR segmentation on single scans and a 4D spatio-temporal setup that segments multiple scans simultaneously while maintaining consistent instance IDs over time. Inputs are LiDAR point clouds with user clicks, and outputs are per-point object masks/instance IDs over 3D or 4D (x, y, z, t) domains. The paper does not explicitly specify task dynamics, attention dynamics, or state dynamics for the model interface.

## Evidence
### Task: Interactive LiDAR segmentation (single scan)
- "the user guides the model to densely label each point in a point cloud through sparse user interactions." (Section I. INTRODUCTION)
- "Depending on whether it operates on a single scan or superimposed consecutive scans, Interactive4D can function as either an LPS or 4D-LPS method." (Section II. RELATED WORK)
- "each point has 3 coordinates (x,y,z)." (Section VI-A. Spatio-Temporal Point Cloud Construction)
- "Given a set of raw clicks  $C_K$  for the K-th iteration" (Section III. METHOD, Click Encoder)
- "the final mask  $\mathcal{M} \in \mathbb{R}^N$  is obtained by applying Softmax over the ID dimension" (Section III. METHOD, Click Fusion)

### Task: Interactive 4D LiDAR segmentation and tracking
- "interactive 4D segmentation, a new paradigm that allows segmenting multiple objects on multiple LiDAR scans simultaneously" (Abstract)
- "superimposing consecutive LiDAR scans within a short temporal window  $[t,t+\tau]$  into a single spatio-temporal point cloud" (Section III. METHOD, Spatio-Temporal Point Cloud)
- "Operating on the 4D volume, it directly provides consistent instance IDs over time" (Abstract)
- "Within each short temporal window  $[t,t+\tau]$  we directly obtain consistent instance IDs by assigning each point to the object with the highest response in  $H_K$" (Section III. METHOD, 4D Inference)
- "Given a set of raw clicks  $C_K$  for the K-th iteration" (Section III. METHOD, Click Encoder)
