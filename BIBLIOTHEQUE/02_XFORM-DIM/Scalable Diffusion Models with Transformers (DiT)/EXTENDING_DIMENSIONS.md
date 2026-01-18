## 1. Basic Metadata

- Title: "Scalable Diffusion Models with Transformers" (Title)
- Authors: "William Peebles\* UC Berkeley Saining Xie New York University" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper claims its primary contribution is: "We explore a new class of diffusion models based on the transformer architecture," "replacing the commonly-used U-Net backbone with a transformer that operates on latent patches." (Abstract)

## 3. Tasks Evaluated

- Task name: Class-conditional image generation on ImageNet 256×256
  - Task type: Generation
  - Dataset(s) used: "ImageNet dataset [31]" (Section 4. Experimental Setup)
  - Domain: "ImageNet dataset [31]" (Section 4. Experimental Setup)
  - Evidence: "We train class-conditional latent DiT models at  $256 \times 256$  and  $512 \times 512$  image resolution on the ImageNet dataset [31], a highly-competitive generative modeling benchmark." (Section 4. Experimental Setup); "Benchmarking class-conditional image generation on ImageNet 256×256." (Table 2 caption)

- Task name: Class-conditional image generation on ImageNet 512×512
  - Task type: Generation
  - Dataset(s) used: "ImageNet" (Section 5.1 State-of-the-Art Diffusion Models)
  - Domain: "ImageNet" (Section 5.1 State-of-the-Art Diffusion Models)
  - Evidence: "512×512 ImageNet. We train a new DiT-XL/2 model on ImageNet at  $512 \times 512$  resolution for 3M iterations with identical hyperparameters as the  $256 \times 256$  model." (Section 5.1 State-of-the-Art Diffusion Models); "Benchmarking class-conditional image generation on ImageNet 512×512." (Table 3 caption)

## 4. Domain and Modality Scope

- Single domain: Yes — "We train class-conditional latent DiT models at  $256 \times 256$  and  $512 \times 512$  image resolution on the ImageNet dataset [31]." (Section 4. Experimental Setup)
- Multiple domains within the same modality: Not claimed.
- Multiple modalities: Not claimed.
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Class-conditional image generation on ImageNet 256×256 | No (trained separately per resolution) | Not specified. | Not specified. | "We train class-conditional latent DiT models at  $256 \times 256$  and  $512 \times 512$  image resolution on the ImageNet dataset [31]." (Section 4. Experimental Setup) |
| Class-conditional image generation on ImageNet 512×512 | No (trained as a new model for this resolution) | Not specified. | Not specified. | "We train a new DiT-XL/2 model on ImageNet at  $512 \times 512$  resolution for 3M iterations with identical hyperparameters as the  $256 \times 256$  model." (Section 5.1 State-of-the-Art Diffusion Models) |

## 6. Input and Representation Constraints

- Input resolution is fixed per experiment: "We train class-conditional latent DiT models at  $256 \times 256$  and  $512 \times 512$  image resolution on the ImageNet dataset [31]." (Section 4. Experimental Setup)
- Latent representation uses a fixed 2D grid with channels: "for  $256 \times 256 \times 3$  images, z has shape  $32 \times 32 \times 4$ ." (Section 3.2 Patchify)
- VAE downsampling constraint: "The VAE encoder has a downsample factor of 8—given an RGB image x with shape  $256 \times 256 \times 3$ , z = E(x) has shape  $32 \times 32 \times 4$ ." (Section 4. Experimental Setup)
- Fixed patch size hyperparameter per model: "Given patch size  $p \times p$ , a spatial representation (the noised latent from the VAE) of shape  $I \times I \times C$  is \"patchified\" into a sequence of length  $T = (I/p)^2$  with hidden dimension d." (Figure 4)
- Patch size choices are discrete: "We add p = 2, 4, 8 to the DiT design space." (Section 3.2 Patchify)
- Token count varies with patch size and input size: "A smaller patch size p results in a longer sequence length and thus more Gflops." (Figure 4)
- Example fixed token count at 512×512: "With a patch size of 2, this XL/2 model processes a total of 1024 tokens after patchifying the  $64 \times 64 \times 4$  input latent." (Section 5.1 State-of-the-Art Diffusion Models)
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "With a patch size of 2, this XL/2 model processes a total of 1024 tokens after patchifying the  $64 \times 64 \times 4$  input latent." (Section 5.1 State-of-the-Art Diffusion Models)
- Sequence length fixed or variable: Variable by design — "sequence of length  $T = (I/p)^2$" and "A smaller patch size p results in a longer sequence length and thus more Gflops." (Figure 4)
- Attention type: Global (standard transformer self-attention) — "we aim to be as faithful to the standard transformer architecture as possible" and the block includes "multi-head self-attention." (Section 3.2 Diffusion Transformer Design Space)
- Mechanisms to manage computational cost: "Training diffusion models directly in high-resolution pixel space can be computationally prohibitive. Latent diffusion models (LDMs) tackle this issue with a two-stage approach" and patch size controls compute: "A smaller patch size p results in a longer sequence length and thus more Gflops." (Section 3.1 Preliminaries; Figure 4)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Absolute (sine-cosine) — "Following patchify, we apply standard ViT frequency-based positional embeddings (the sine-cosine version) to all input tokens." (Section 3.2 Patchify)
- Where it is applied: Input tokens only — "we apply ... positional embeddings ... to all input tokens." (Section 3.2 Patchify)
- Fixed across experiments vs. modified/ablated: Not specified; only one positional encoding mechanism is described. (Section 3.2 Patchify)

## 9. Positional Encoding as a Variable

- Treated as a fixed architectural assumption: "Following patchify, we apply standard ViT frequency-based positional embeddings (the sine-cosine version) to all input tokens." (Section 3.2 Patchify)
- Multiple positional encodings compared: Not specified.
- PE choice claimed as not critical/secondary: Not specified.

## 10. Evidence of Constraint Masking

- Model sizes / compute scale: "They cover a wide range of model sizes and flop allocations, from 0.3 to 118.6 Gflops, allowing us to gauge scaling performance." (Section 3.2 Model size)
- Dataset size(s): Dataset size not specified; dataset named only as "ImageNet dataset [31]." (Section 4. Experimental Setup)
- Performance gains attributed to scaling model size/tokens: "We find that DiTs with higher Gflops—through increased transformer depth/width or increased number of input tokens—consistently have lower FID." (Abstract)
- Additional evidence for scaling: "increasing model size and decreasing patch size yields considerably improved diffusion models." (Section 5 Experiments)
- Scaling sampling compute does not replace model scale: "scaling-up sampling compute cannot compensate for a lack of model compute." (Section 5.2 Scaling Model vs. Sampling Compute)

## 11. Architectural Workarounds

- Latent diffusion to reduce compute: "Training diffusion models directly in high-resolution pixel space can be computationally prohibitive. Latent diffusion models (LDMs) tackle this issue with a two-stage approach" (Section 3.1 Preliminaries).
- Patchify with adjustable patch size to control sequence length and compute: "Given patch size  $p \times p$ ... \"patchified\" into a sequence of length  $T = (I/p)^2$" and "A smaller patch size p results in a longer sequence length and thus more Gflops." (Figure 4)
- Compute-efficient conditioning block: "adaLN adds the least Gflops and is thus the most compute-efficient." (Section 3.2 Diffusion Transformer Design Space)

## 12. Explicit Limitations and Non-Claims

- Future work / limitations: "future work should continue to scale DiTs to larger models and token counts." (Section 6. Conclusion)
- Future work / scope expansion: "DiT could also be explored as a drop-in backbone for text-to-image models like DALL·E 2 and Stable Diffusion." (Section 6. Conclusion)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single ImageNet image domain — "We train class-conditional latent DiT models at  $256 \times 256$  and  $512 \times 512$  image resolution on the ImageNet dataset [31]." (Section 4. Experimental Setup)
> – Task structure: Class-conditional image generation at fixed resolutions — "Benchmarking class-conditional image generation on ImageNet 256×256." and "Benchmarking class-conditional image generation on ImageNet 512×512." (Table 2 caption; Table 3 caption)
> – Representation rigidity: Fixed latent grid and patchified tokens — "z has shape  $32 \times 32 \times 4$ " and "sequence of length  $T = (I/p)^2$" (Section 3.2 Patchify; Figure 4)
> – Model sharing vs specialization: Separate models per resolution — "We train a new DiT-XL/2 model on ImageNet at  $512 \times 512$  resolution" (Section 5.1 State-of-the-Art Diffusion Models)
> – Role of positional encoding: Fixed sine-cosine PE applied to inputs — "we apply standard ViT frequency-based positional embeddings (the sine-cosine version) to all input tokens." (Section 3.2 Patchify)

### 14. Final Classification

**Single-task, single-domain**

The evaluation is limited to class-conditional image generation on ImageNet at two fixed resolutions: "We train class-conditional latent DiT models at  $256 \times 256$  and  $512 \times 512$  image resolution on the ImageNet dataset [31]." (Section 4. Experimental Setup) The paper also trains separate models per resolution ("We train a new DiT-XL/2 model on ImageNet at  $512 \times 512$  resolution"), with no evidence of multi-domain or multi-task training. (Section 5.1 State-of-the-Art Diffusion Models)
