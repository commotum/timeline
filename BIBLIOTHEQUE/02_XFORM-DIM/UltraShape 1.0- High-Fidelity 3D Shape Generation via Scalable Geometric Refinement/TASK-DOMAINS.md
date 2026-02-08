# UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement (2025)
Source: UltraShape 1.0- High-Fidelity 3D Shape Generation via Scalable Geometric Refinement.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Watertight geometry processing | 3D meshes | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Watertight geometry (mesh/SDF field) | 3D (x, y, z) (inferred) | Capped (inferred) |
| Image-conditioned 3D geometry generation | Condition images; coarse-shape voxel queries/latent tokens | 2D (x, y); 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Generated 3D geometry/mesh | 3D (x, y, z) (inferred) | Capped (inferred) |
| VAE 3D geometry reconstruction | Surface points; latent shape tokens | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Reconstructed 3D geometry (SDF/mesh) | 3D (x, y, z) (inferred) | Capped (inferred) |
| Training-free 3D stylization | First-stage condition image; second-stage condition image; coarse voxel representation | 2D (x, y); 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Stylized 3D geometry/mesh | 3D (x, y, z) (inferred) | Capped (inferred) |

## Summary
UltraShape 1.0 covers four task intents in the OCR text: watertight 3D geometry processing, image-conditioned 3D geometry generation, VAE-based 3D reconstruction, and training-free 3D stylization. Inputs span 3D geometric objects and 2D image conditions, while outputs are consistently 3D geometry, so the dimensional range is 2D and 3D at input and 3D at output. The interface behavior is capped by explicit voxel resolutions and bounded token/image settings rather than open-ended streaming. Attention behavior is not explicitly framed with the glossary labels, but the described token-based self/cross-attention setup supports a static-attention interpretation, and the latent refinement/reconstruction pipelines support a constructed-state interpretation.

## Evidence
### Task: Watertight geometry processing
- "we develop a novel voxel-based reconstruction approach for watertight geometry processing." (Section 2.1 Data Curation Pipeline)
- "The method operates in a sparse volumetric domain, where topological ambiguities can be resolved robustly before surface extraction." (Section 2.1 Data Curation Pipeline)
- "CUDA-parallel sparse data structures and algorithms, enabling scalable voxel reconstruction at resolutions up to 2048^3." (Section 2.1 Data Curation Pipeline)
- Inference: `3D (x, y, z)` is inferred from the "sparse volumetric domain" and voxel reconstruction wording; `Capped` is inferred from the explicit upper resolution ("up to 2048^3"); `Static` attention and `Constructed` state are inferred because the process is described as structured voxel reconstruction with internally constructed volumetric representation, not runtime retrieval/open interaction.

### Task: Image-conditioned 3D geometry generation
- "we introduce UltraShape 1.0, a scalable 3D diffusion framework for high-fidelity 3D geometry generation." (Abstract)
- "we first generate a coarse representation that captures the overall shape of the object, and then refine it using voxel-based queries to synthesize detailed, high-quality geometry." (Section 2.2 Geometry Generation)
- "The refinement stage employs a DiT architecture with self-attention over latent tokens, and the results are shown in Fig. 3. Spatial information is injected via RoPE at each layer, while image conditioning is incorporated through cross-attention using DINOv2 [16] features." (Section 2.2 Geometry Generation)
- "We use a voxel resolution of 128 for both training and inference, and adopt a progressive multi-stage strategy that jointly increases token count and image resolution: (1) 4096 tokens at 518 resolution for 10K steps; (2) 8192 tokens at 1022 resolution for 15K steps; and (3) 10240 tokens at 1022 resolution for 60K steps." (Section 3.1 Implementation Details)
- Inference: Input dimension `2D (x, y); 3D (x, y, z)` is inferred from condition images plus voxel queries; `Capped` dynamics is inferred from fixed/bounded voxel, token, and image-resolution settings; `Static` attention is inferred because attention operates over provided tokens rather than runtime retrieval; `Constructed` state is inferred from iterative denoising and latent refinement before decoding.

### Task: VAE 3D geometry reconstruction
- "These surface points are used as inputs to the VAE encoder." (Section 3.1 Implementation Details)
- "SDF values are computed for all supervision points and used to define the reconstruction loss." (Section 3.1 Implementation Details)
- "As the number of latent tokens increases, reconstruction quality consistently improves, demonstrating strong potential to reconstruct high-fidelity geometries." (Section 3.3 Shape Generation)
- "Figure 9 Comparison of VAE inference results when extrapolating the number of tokens during reconstruction." (Figure 9)
- Inference: `3D (x, y, z)` input/output is inferred from surface-point/SDF/geometry reconstruction descriptions; `Capped` dynamics is inferred from explicit finite sampling and token settings; `Static` attention is inferred because no runtime retrieval/selection policy is described; `Constructed` state is inferred from encoding geometry into latent tokens before decoding.

### Task: Training-free 3D stylization
- "Training-Free Stylization. We further discovered the potential of training-free stylization using voxel-conditioned latent." (Section 2.2 Geometry Generation)
- "by conditioning on different images in the two stages, we can generate 3D geometry that follows the coarse shape from the image used in the first stage and finer stylized details from the image used in the second stage" (Section 2.2 Geometry Generation)
- "the coarse voxel representation enables the second stage to perform fine-detail sculpting without introducing conflicts." (Section 2.2 Geometry Generation)
- Inference: Input dimension `2D (x, y); 3D (x, y, z)` is inferred from two image conditions plus coarse voxel representation; `Capped` dynamics follows the same bounded token/resolution interface as generation; `Static` attention and `Constructed` state are inferred from the same two-stage latent refinement mechanism described for the generation pipeline.
