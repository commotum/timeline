# Scaling Transformer-Based Novel View Synthesis with Models Token Disentanglement and Synthetic Data (Year not specified in the paper)
Source: Scaling Transformer-Based Novel View Synthesis with Models Token Disentanglement and Synthetic Data.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Novel view synthesis (generation of unseen scene viewpoints) | Sparse source-view images with source/target Plucker ray-coordinate patches | 3D (x, y, z) or (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Target novel-view RGB image | 2D (x, y) | Capped (inferred) |

## Summary
The paper covers one core task: feed-forward novel view synthesis from sparse multi-view image observations. The model consumes a bounded set of source views plus ray/camera geometry inputs and reconstructs a target RGB view. The OCR supports a multi-view spatial input domain with 2D image outputs, and bounded (non-open) input/output sizing in the reported setup. Attention and state-type labels are inferred from the feed-forward transformer formulation over a provided token set.

## Evidence
### Task: Novel view synthesis (generation of unseen scene viewpoints)
- "Novel view synthesis (NVS) [19, 26] is a well-studied and important problem in computer vision, where the task is to generate unseen perspectives of a scene from a given set of images." (Section 1. Introduction)
- "Our method performs feed-forward novel-view synthesis from a series of input images, such as the pairs shown above." (Figure 1. Overview)
- "Finally, the transformer network is trained to reconstruct the target output tokens  $O_i^t$  from the Plücker patch embeddings." (Section 3. Background)
- "The target patches are unpatchified to get the target image  $T \in R^{H \times W \times 3}$  (see Figure 2)." (Section 3. Background)
- "for scene-level synthesis, we follow LVSM and train using 2 input views and test using 6 target views fed one at a time." (Section 5.1. Implementation Details)
- Inference: `In Dimension` is marked `3D (x, y, z) or (x, y, t) (inferred)` because the task uses a "series of input images" (Figure 1) and multi-view source/target setup; `In Dynamics`/`Out Dynamics` are marked `Capped (inferred)` because experiments use explicit finite view counts (e.g., 2 source views, 6 targets, one target at a time in Section 5.1; 2/4/8 source-view settings in Section 5.5); `Attention Dynamic` is `Static (inferred)` from the fixed-token self-attention formulation "SelfAttn_l([x_l^s, x_l^t])" (Section 3); `State Dynamic` is `Direct (inferred)` because the paper describes a feed-forward mapping from provided inputs to reconstructed target views, without a persistent external memory/state mechanism.
