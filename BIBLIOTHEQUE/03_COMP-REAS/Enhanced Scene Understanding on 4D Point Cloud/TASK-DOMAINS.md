# X4D-SceneFormer: Enhanced Scene Understanding on 4D Point Cloud Videos through Cross-Modal Knowledge Transfer (2024)
Source: Enhanced Scene Understanding on 4D Point Cloud.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4D semantic segmentation | 4D point cloud video (T-frame point cloud sequence) | 4D (x, y, z, t) | Not specified in the paper. | Static (inferred) | Not specified in the paper. | Point-level semantic labels per frame | 4D (x, y, z, t) | Not specified in the paper. |
| 4D action segmentation | 4D point cloud video (T-frame point cloud sequence) | 4D (x, y, z, t) | Not specified in the paper. | Static (inferred) | Not specified in the paper. | Frame-level action labels | 1D (t) | Not specified in the paper. |
| 4D action recognition | 4D point cloud video (T-frame point cloud sequence) | 4D (x, y, z, t) | Not specified in the paper. | Static (inferred) | Not specified in the paper. | Single action label for whole video | 0D | Fixed |

## Summary
The paper defines three 4D point cloud video tasks: semantic segmentation, action segmentation, and action recognition. Inputs are spatiotemporal 4D (x, y, z, t) point cloud sequences, while outputs range from per-point labels (4D) to per-frame labels (1D) to a single video-level label (0D). Attention is described via self-attention across the full sequence, supporting a Static attention classification (inferred), while dynamics and state are otherwise not specified.

## Evidence
### Task: 4D semantic segmentation
- "a point cloud video consisting of T frames with N points as input" (Problem Formulation)
- "$$SemSeg: \mathbb{R}^{T \times N \times 3} \mapsto \mathbb{R}^{T \times N}, \tag{1}$$" (Problem Formulation)
- "the former two segmentation tasks perform classification on point and frame levels respectively" (Problem Formulation)
- Inference: Attention Dynamic = Static (inferred) because "several selfattention layers are applied to extract the sequential information across the sequence dimension." (4D Point Cloud Architecture)

### Task: 4D action segmentation
- "a point cloud video consisting of T frames with N points as input" (Problem Formulation)
- "$$ActionSeg: \mathbb{R}^{T \times N \times 3} \mapsto \mathbb{R}^{T}, \tag{2}$$" (Problem Formulation)
- "the former two segmentation tasks perform classification on point and frame levels respectively" (Problem Formulation)
- Inference: Attention Dynamic = Static (inferred) because "several selfattention layers are applied to extract the sequential information across the sequence dimension." (4D Point Cloud Architecture)

### Task: 4D action recognition
- "a point cloud video consisting of T frames with N points as input" (Problem Formulation)
- "$$\mathbb{R}^{T \times N \times 3} \mapsto \mathbb{R}^1$$" (Problem Formulation)
- "the recognition task identify single action for the whole video." (Problem Formulation)
- Inference: Attention Dynamic = Static (inferred) because "several selfattention layers are applied to extract the sequential information across the sequence dimension." (4D Point Cloud Architecture)

---

## CSV Output (required)
CSV written to `.TASK-DOMAINS.csv.tmp.fbce656d76484026a9e820bd3631487d`.
