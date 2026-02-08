# Generalizable Neural Radiance Fields for Novel View Synthesis with Transformer (Not specified in the paper)
Source: TransNeRF- Generalizable Neural Radiance Fields for Novel View Synthesis with Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Novel view synthesis | Multi-view source images with camera parameters; query 3D points and viewing directions | 2D (x, y); 3D (x, y, z) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Novel-view rendered RGB image | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper covers one task: novel view synthesis from multi-view source observations. Inputs combine 2D source-view image information with 3D query geometry (point locations and ray directions), and outputs are rendered 2D novel-view RGB images. The model is described as accepting an arbitrary number of source views, while each render is a finite image over camera rays. From the architecture description, the attention policy is classified as Static and the state as Constructed (both inferred).

## Evidence
### Task: Novel view synthesis
- "Abstract—We **Transformer-based** NeRF propose a (TransNeRF) to learn a generic neural radiance field conditioned on observed-view images for the novel view synthesis task." (Section Abstract)
- "Given captured multi-view images  $\{\mathbf{I}^m\}_{m=1}^M$  (M source views) of diverse scenes and their camera parameters  $\{\boldsymbol{\Theta}^m\}_{m=1}^M$  (camera poses, intrinsic parameters and scene bounds), TransNeRF reconstructs a generic radiance field" (Section III. METHODOLOGY)
- "where (x, y, z) is a 3D point location, **d** denotes a unit-length viewing ray-direction and outputs are a differential volumetric density  $\sigma$  and a directional emitted color **c**." (Section III. METHODOLOGY)
- "As in NeRF [1], classical volume rendering [32], a differentiable ray marching rendering, is then utilized to render a projection 2D image from our radiance field scene representation" (Section III. METHODOLOGY)
- "Our Density-ViewDecoder is invariant to permutations of source views and can receive an arbitrary number of source views." (Section III-A. Density Decoder in Surrounding-view Space)
- Inference: `2D (x, y); 3D (x, y, z)` for In Dimension is inferred from the paper describing projected 2D pixels/images and 3D query points/rays; `Open` for In Dynamics is inferred from "arbitrary number of source views"; `Static` for Attention Dynamic is inferred because attention layers (`MH-Attn`) operate over the provided source-view/ray neighborhoods rather than runtime retrieval outside the given input set; `Constructed` for State Dynamic is inferred from explicit latent density/color representations and a learned radiance-field representation; `2D (x, y)` and `Capped` for output are inferred from rendering finite target-view images over a finite set of camera rays and stated image resolutions (Sections III and IV-A).
