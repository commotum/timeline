## 1. Basic Metadata
- Title: "UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement" (title block)
- Authors: "Tanghui Jia<sup>\*1</sup>, Dongyu Yan<sup>\*2</sup>, Dehao Hao<sup>\*3</sup>, Yang Li<sup>2</sup>, Kaiyi Zhang<sup>3</sup>, Xianyi He<sup>1</sup>, Lanjiong Li<sup>2</sup> Yuhan Wang<sup>5</sup>, Jinnan Chen<sup>4</sup>, Lutao Jiang<sup>2</sup>, Qishen Yin<sup>1</sup>, Long Quan<sup>3</sup>, Ying-Cong Chen<sup>2</sup>, Li Yuan<sup>1</sup>" (title block)
- Year: 2025 ("Date: December 29, 2025" (front matter))
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
The paper introduces a two-stage 3D diffusion framework for high-fidelity 3D geometry generation that refines coarse structures with voxel queries and RoPE-based spatial anchors ("we introduce UltraShape 1.0, a scalable 3D diffusion framework for high-fidelity 3D geometry generation" and "performing voxel-based refinement at fixed spatial locations, where voxel queries derived from coarse geometry provide explicit positional anchors encoded via RoPE", Abstract).

## 3. Tasks Evaluated
Task 1
- Task name: Data watertightening / watertight geometry processing.
- Task type: Other (watertight geometry processing/remeshing).
- Dataset(s) used: Objaverse 3D models ("Our initial data pool comes from Objaverse [4], which contains about 800K 3D models across diverse categories and styles."; "refined the initial 800K models down to approximately 330K valid samples, of which 120K were identified as high-quality." (2.1 Data Curation Pipeline)).
- Domain: 3D models/meshes ("3D models" in Objaverse (2.1 Data Curation Pipeline)).
- Quotes: "## 3.2 Data Watertightening" (3.2 Data Watertightening); "we develop a novel voxel-based reconstruction approach for watertight geometry processing." (2.1 Data Curation Pipeline).

Task 2
- Task name: 3D shape/geometry generation (image-conditioned).
- Task type: Generation.
- Dataset(s) used: Objaverse 120K filtered samples and rendered images ("using 120K filtered samples from Objaverse."; "For each object, we render 16 images" (3.1 Implementation Details)).
- Domain: 3D geometry with image conditioning ("image-conditioned 3D generation" (3.1 Implementation Details)).
- Quotes: "we introduce UltraShape 1.0, a scalable 3D diffusion framework for high-fidelity 3D geometry generation." (Abstract); "## 3.3 Shape Generation" (3.3 Shape Generation); "We then evaluate the generation performance of our model" (3.3 Shape Generation).

Task 3
- Task name: VAE reconstruction (token extrapolation).
- Task type: Reconstruction.
- Dataset(s) used: Not specified.
- Domain: 3D geometry/shape reconstruction.
- Quotes: "Figure 9 Comparison of VAE inference results when extrapolating the number of tokens during reconstruction." (Figure 9); "As the number of latent tokens increases, reconstruction quality consistently improves" (3.3 Shape Generation).

Task 4
- Task name: Training-free 3D stylization (image-conditioned).
- Task type: Other (stylization).
- Dataset(s) used: Not specified.
- Domain: 3D geometry with image conditioning.
- Quotes: "Training-Free Stylization. We further discovered the potential of training-free stylization using voxel-conditioned latent." (2.2 Geometry Generation); "by conditioning on different images in the two stages, we can generate 3D geometry that follows the coarse shape from the image used in the first stage and finer stylized details from the image used in the second stage" (2.2 Geometry Generation).

## 4. Domain and Modality Scope
- Single domain: Yes; 3D object shapes from Objaverse ("800K 3D models across diverse categories and styles." (2.1 Data Curation Pipeline)).
- Multiple domains within the same modality: Not explicitly stated; categories/styles are mentioned within 3D models ("across diverse categories and styles" (2.1 Data Curation Pipeline)).
- Multiple modalities: Yes; images are used for conditioning 3D generation ("For each object, we render 16 images" (3.1 Implementation Details); "image conditioning is incorporated through cross-attention using DINOv2 [16] features." (2.2 Geometry Generation)).
- Domain generalization or cross-domain transfer: Generalization across object categories is claimed ("robust generalization across diverse object categories." (2.2 Geometry Generation)); cross-domain transfer not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Data watertightening / watertight geometry processing | Not specified | Not specified | Not specified | "data processing pipeline that includes a novel watertight processing method" (Abstract); "novel voxel-based reconstruction approach for watertight geometry processing." (2.1 Data Curation Pipeline) |
| 3D shape/geometry generation (image-conditioned) | Not specified | Yes (refinement DiT fine-tuned) | Not specified | "The diffusion transformer (DiT) for geometry refinement is also initialized from Hunyuan3D-2.1 and fine-tuned on our dataset." (3.1 Implementation Details) |
| VAE reconstruction (token extrapolation) | Not specified | Yes (VAE fine-tuned) | Not specified | "The VAE used in the refinement stage is initialized from the Hunyuan3D-2.1 VAE and fine-tuned for 55K steps" (3.1 Implementation Details) |
| Training-free 3D stylization (image-conditioned) | Not specified | No (training-free) | Not specified | "Training-Free Stylization. We further discovered the potential of training-free stylization using voxel-conditioned latent." (2.2 Geometry Generation) |

## 6. Input and Representation Constraints
- Fixed/variable input resolution: "All images are rendered at a resolution of  $1024^2$ ." (3.1 Implementation Details); "(1) 4096 tokens at 518 resolution ... (2) 8192 tokens at 1022 resolution ... (3) 10240 tokens at 1022 resolution." and "During inference, we use 32768 tokens and an image resolution of 1022" (3.1 Implementation Details).
- Fixed spatial grid: "voxel-based refinement on voxel queries defined over a fixed-resolution grid" (2.2 Geometry Generation); "We use a voxel resolution of 128 for both training and inference" (3.1 Implementation Details).
- Fixed number of tokens (variable across stages): "4096 tokens"; "8192 tokens"; "10240 tokens"; "During inference, we use 32768 tokens" (3.1 Implementation Details).
- Fixed sampling counts for VAE input/supervision: "we sample approximately 600K surface points for VAE input and 1M points for supervision." (3.1 Implementation Details).
- Bounded query perturbations: "uniform query perturbation sampled from [-1/128, 1/128]." (3.1 Implementation Details).
- Fixed dimensionality/volumetric assumption: "The method operates in a sparse volumetric domain" (2.1 Data Curation Pipeline).
- Fixed patch size: Not specified.
- Padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: "During inference, we use 32768 tokens" (3.1 Implementation Details).
- Fixed or variable sequence length: Variable token counts across stages ("4096 tokens"; "8192 tokens"; "10240 tokens"; "32768 tokens" (3.1 Implementation Details)).
- Attention type: Global self-attention over tokens and cross-attention for conditioning ("The refinement stage employs a DiT architecture with self-attention over latent tokens"; "image conditioning is incorporated through cross-attention using DINOv2 [16] features." (2.2 Geometry Generation)).
- Mechanisms to manage computational cost: Two-stage coarse-to-fine pipeline ("two-stage coarse-to-fine design" (2.2 Geometry Generation)); progressive scaling of tokens/resolution ("progressive multi-stage strategy that jointly increases token count and image resolution" (3.1 Implementation Details)); token masking ("An image token masking strategy is applied to suppress irrelevant background information" (2.2 Geometry Generation)).

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: RoPE ("coordinates are encoded using rotary positional embeddings (RoPE) [19]." (2.2 Geometry Generation); "explicit positional anchors encoded via RoPE" (Abstract)).
- Where it is applied: At each layer for spatial information ("Spatial information is injected via RoPE at each layer" (2.2 Geometry Generation)).
- Fixed/modified/ablated: Not specified; no ablations or alternative positional encodings are described.

## 9. Positional Encoding as a Variable
- Core research variable or fixed assumption: Fixed architectural component ("coordinates are encoded using rotary positional embeddings (RoPE)" (2.2 Geometry Generation)).
- Multiple positional encodings compared: Not specified.
- PE described as "not critical" or secondary: Not specified.

## 10. Evidence of Constraint Masking
- Dataset size(s): "Our initial data pool comes from Objaverse [4], which contains about 800K 3D models"; "refined the initial 800K models down to approximately 330K valid samples, of which 120K were identified as high-quality." (2.1 Data Curation Pipeline); "using 120K filtered samples from Objaverse." (3.1 Implementation Details).
- Model size(s): Parameter counts not specified; token budgets are explicitly scaled ("4096 tokens"; "8192 tokens"; "10240 tokens"; "32768 tokens" (3.1 Implementation Details)).
- Performance gains attributed to scaling tokens: "As the number of latent tokens increases, reconstruction quality consistently improves" and "generalizes well to significantly larger token counts at test time, producing substantially improved geometric details" (3.3 Shape Generation).
- Performance gains attributed to architectural hierarchy: "two-stage coarse-to-fine strategy" and "decouples spatial localization from geometric detail synthesis" to enable "fine-grained geometry generation at scale." (Abstract).
- Training tricks/strategies: "progressive multi-stage strategy that jointly increases token count and image resolution" (3.1 Implementation Details).

## 11. Architectural Workarounds
- Two-stage coarse-to-fine pipeline: "a two-stage coarse-to-fine design" where "a coarse global structure is first generated and then refined" (2.2 Geometry Generation).
- Hybrid representations: "DiT-based 3D generation model operating on a vector set representation as the first-stage generator" and refinement via "voxel-based refinement on voxel queries defined over a fixed-resolution grid" (2.2 Geometry Generation).
- Spatial decoupling and structured queries: "decouple spatial localization from geometric detail synthesis" using fixed voxel queries (2.2 Geometry Generation).
- RoPE-based spatial anchors: "coordinates are encoded using rotary positional embeddings (RoPE)" and "Spatial information is injected via RoPE at each layer" (2.2 Geometry Generation).
- Off-surface decoding and perturbations: "shape VAE is extended to decode geometry at offsurface locations" and "surface queries are augmented with bounded spatial perturbations" (2.2 Geometry Generation).
- Conditioning and masking: "image conditioning is incorporated through cross-attention using DINOv2 [16] features" with "image token masking" (2.2 Geometry Generation).
- Sparse voxel infrastructure for scalability: "CUDA-parallel sparse data structures and algorithms, enabling scalable voxel reconstruction at resolutions up to 2048^3." (2.1 Data Curation Pipeline).

## 12. Explicit Limitations and Non-Claims
- Limitations or future work: Not specified.
- Explicit non-claims (e.g., open-world or unrestrained multi-task learning): Not specified.

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> – Domain scope: Single 3D object domain from Objaverse with image conditioning; no cross-domain transfer claims.
> – Task structure: Data watertightening plus two-stage 3D generation/reconstruction/stylization within the same 3D domain.
> – Representation rigidity: Fixed-resolution voxel grid (128) with fixed/variable token counts (4096–32768) and fixed image resolutions (1024^2, 1022).
> – Model sharing vs specialization: Separate data-processing pipeline; coarse stage uses Hunyuan3D-2.1, refinement DiT/VAE fine-tuned.
> – Role of positional encoding: RoPE provides explicit spatial anchors at each layer as a fixed architectural component.

### 14. Final Classification
**Multi-task, single-domain.** The paper evaluates "Data Watertightening" and "Shape Generation" on 3D object data ("800K 3D models across diverse categories and styles") with image-conditioned generation, keeping evaluation within a single 3D domain. It claims generalization only "across diverse object categories" and does not claim cross-domain transfer.
