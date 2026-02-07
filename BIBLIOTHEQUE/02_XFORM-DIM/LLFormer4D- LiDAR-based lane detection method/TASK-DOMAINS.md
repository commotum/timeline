# LLFormer4D: LiDAR-based lane detection method by temporal feature fusion and sparse transformer (2025)
Source: LLFormer4D- LiDAR-based lane detection method.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiDAR lane detection | LiDAR point clouds (multi-frame) | 4D (x, y, z, t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | lane key points; lane curve parameters | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper presents a single task: LiDAR lane detection from multi-frame point clouds, producing lane key points and fitted lane curve parameters. The input is treated as spatiotemporal (4D) and bounded in extent via temporal fusion and FOV cropping (both inferred), with outputs capped by an assumed maximum number of lanes (inferred). The model uses LKQ-based cross-attention and reference-point updates, supporting dynamic attention and constructed state (both inferred from the described mechanisms).

## Evidence
### Task: LiDAR lane detection
- "Lane detection is a fundamental problem in autonomous driving" (Abstract)
- "The inputs of our model consist of points." (Section 4.3 Implementation details)
- "Temporal Feature Fusion module is introduced to enhance accuracy and robustness by integrating features from multi-frame point clouds." (Abstract)
- "Finally, the algorithm utilises the learnt LKQ to classify and regress the lane key points." (Figure 2)
- "Finally, the lane curve parameters are fitted from the lane key points" (Section 3.1 Overview of the proposed network)
- "The 2D Lane reference-point is encoded to a 256-dimensional vector via MLP" (Figure 4)
- "we utilise hLKQ to learn features of lane key points row by row with h representing the assumed maximum number of lanes." (Section 3.4)
- "interacts with the LKQ (based on cross-attention)." (Figure 2)
- "the lane key points detected and output by the network are stored and used to update the reference points." (Section 4.4.2)
- "crop the feature map based on the field of view (FOV) in the current frame" (Section 3.3 Temporal Feature Fusion module)
- Inference: In Dimension as 4D (x, y, z, t) because the method integrates multi-frame point clouds; In Dynamics as Capped because the fusion uses fixed historical frames and FOV cropping; Attention Dynamic as Dynamic due to LKQ cross-attention; State Dynamic as Constructed because reference points are updated from detected key points; Out Dimension as 2D (x, y) based on 2D lane reference points; Out Dynamics as Capped due to an assumed maximum number of lanes. (Abstract; Figure 2; Figure 4; Section 3.3; Section 3.4; Section 4.4.2)
