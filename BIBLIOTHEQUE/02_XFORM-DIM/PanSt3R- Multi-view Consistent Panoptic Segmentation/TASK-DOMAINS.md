# PanSt3R: Multi-view Consistent Panoptic Segmentation (Not specified in the paper.)
Source: PanSt3R- Multi-view Consistent Panoptic Segmentation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D reconstruction | multi-view RGB images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | 3D point-maps / 3D geometry (per-pixel 3D points) | 3D (x, y, z) | Not specified in the paper. |
| panoptic segmentation (semantic + instance) | multi-view RGB images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | semantic segmentation masks; instance segmentation masks (panoptic labels) | 2D (x, y) | Capped (inferred) |

## Summary
PanSt3R takes a set of multi-view RGB images and jointly outputs per-pixel 3D point-maps and panoptic labels (semantic classes and instance IDs). The paper explicitly describes 2D image inputs, 3D geometry outputs for reconstruction, and 2D mask outputs for panoptic segmentation, with instance counts capped by a maximum m. Attention and state dynamics are inferred as Static and Constructed based on cross-attention over fixed frame tokens and MUSt3R's internal memory. Input and reconstruction output size dynamics are otherwise not specified in the paper.

## Evidence
### Task: 3D reconstruction
- "we aim to jointly perform 3D reconstruction and panoptic segmentation" (Section 3, Problem statement)
- "3D point-maps  $\mathbf{X} \in \mathbb{R}^{N \times W \times H \times 3}$" (Section 3, Problem statement)
- Inference: Attention Dynamic marked Static and State Dynamic marked Constructed because the model uses a "mask transformer DEC<sup>P</sup> that attends to multi-view frame tokens  $\{\mathbf{f}_n\}$  using cross-attention" and MUSt3R "maintaining an internal memory of the previously seen images" (Section 3.1).

### Task: panoptic segmentation (semantic + instance)
- "we aim to jointly perform 3D reconstruction and panoptic segmentation" (Section 3, Problem statement)
- "semantic segmentation masks  $\mathbf{M}^{\text{CLS}} \in \{1 \dots C\}^{N \times W \times H}$" (Section 3, Problem statement)
- "instance segmentation masks  $\mathbf{M}^{\text{INST}} \in \{1 \dots m\}^{N \times W \times H}$" (Section 3, Problem statement)
- Inference: Out Dynamics marked Capped because the paper specifies "M the maximum number of instances M in the scene." Attention Dynamic marked Static and State Dynamic marked Constructed because the model uses a "mask transformer DEC<sup>P</sup> that attends to multi-view frame tokens  $\{\mathbf{f}_n\}$  using cross-attention" and MUSt3R "maintaining an internal memory of the previously seen images" (Section 3.1).

---

## CSV Output (required)
