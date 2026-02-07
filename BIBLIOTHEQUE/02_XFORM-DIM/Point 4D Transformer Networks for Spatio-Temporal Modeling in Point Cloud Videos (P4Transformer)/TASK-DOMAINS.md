# Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos (Not specified in the paper)
Source: Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos (P4Transformer).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (3D action recognition) | point cloud video clip | 4D (x, y, z, t) | Fixed | Static (inferred) | Direct (inferred) | action predictions (video-level labels) | 0D (inferred) | Fixed (inferred) |
| segmentation (4D semantic segmentation) | point cloud video clip | 4D (x, y, z, t) | Fixed (inferred) | Static (inferred) | Direct (inferred) | point predictions (semantic labels per point) | 4D (x, y, z, t) | Fixed (inferred) |

## Summary
The paper evaluates P4Transformer on two point cloud video tasks: video-level classification for 3D action recognition and point-level semantic segmentation. Inputs are spatiotemporal point cloud videos with 4D coordinates (x, y, z, t), while outputs are either video-level action labels or per-point semantic labels. Action recognition uses fixed-size clips with fixed point sampling, and segmentation uses fixed-length clips (fixed dynamics inferred); attention is global self-attention over the embedded clip (static, inferred) with a direct feedforward mapping (state dynamic inferred).

## Evidence
### Task: classification (3D action recognition)
- "a video-level classification task, *i.e.*, 3D action recognition" (Section 1. Introduction)
- "given a point cloud video" (Section 3.3)
- "Point cloud videos are split into multiple clips (with a fixed number of frames) as inputs." (Section 4.1)
- "we sample 2,048 points for each frame." (Section 4.1)
- "self-attention blocks) are stacked to capture appearance and motion information across all encoded local features." (Section 3.3)
- "an MLP layer converts the global feature to action predictions." (Section 3.3)
- Inference: Labeled Attention Dynamic as Static, State Dynamic as Direct, and Output Dimension/Out Dynamics as 0D Fixed because the model applies self-attention over all encoded local features and maps a pooled global feature to action predictions for video-level classification. (Section 3.3)

### Task: segmentation (4D semantic segmentation)
- "a point-level prediction task, *i.e.*, 4D semantic segmentation." (Section 1. Introduction)
- "The 4D semantic segmentation can be seen as a point-level classification task." (Section 3.3)
- "we conduct experiments on video clips with length of 3 frames." (Section 4.2)
- "anchor coordinates, *i.e.*, (x, y, z, t)" (Section 3.2.1)
- "an MLP layer that converts point features to point predictions." (Section 3.3)
- Inference: Marked In/Out Dynamics as Fixed and Attention/State as Static/Direct because experiments use fixed-length clips and the transformer applies self-attention over all encoded local features before producing point predictions. (Sections 4.2 and 3.3)
