# Hierarchical Text-Conditional Image Generation with CLIP Latents (Not specified in the paper)
Source: Hierarchical Text-Conditional Image Generation with CLIP Latents.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text-conditional image generation | text captions | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |
| Image super-resolution (upsampling) | low-resolution images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | higher-resolution images | 2D (x, y) (inferred) | Fixed (inferred) |
| Image variation generation | image | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | image variations | 2D (x, y) (inferred) | Fixed (inferred) |
| Image interpolation / blending | two images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | interpolated images | 2D (x, y) (inferred) | Fixed (inferred) |
| Text-guided image manipulation (text diffs) | image; text description | 2D (x, y) (inferred); 1D (t) (inferred) | Fixed (inferred); Capped (inferred) | Static (inferred) | Constructed (inferred) | edited image | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper presents unCLIP, a text-conditional image generation system that maps captions to images and uses diffusion upsamplers for higher-resolution outputs. It also enables image-to-image variations, interpolation between two images, and language-guided image manipulation via text diffs. Across these tasks, inputs span 1D text and 2D image grids with capped text context and fixed image resolutions, and the model operates with static attention over provided inputs and constructed latent state (all inferred from the described architecture).

## Evidence
### Task: Text-conditional image generation
- "text-to-image generation process: a CLIP text embedding is first fed to an autoregressive or diffusion prior to produce an image embedding" (Figure 2 caption)
- "and then this embedding is used to condition a diffusion decoder which produces a final image." (Figure 2 caption)
- "Text encoder context 256" (Table 3, Training Details)
- "upsample images from  $64 \times 64$  to  $256 \times 256$  resolution" (Section 2.1 Decoder)
- Inference: In/Out Dimensions and Dynamics labeled as 1D text and 2D images with capped text length and fixed resolutions based on the text encoder context and fixed upsampling sizes; Attention Static and State Constructed inferred from conditioning on text and intermediate embeddings.

### Task: Image super-resolution (upsampling)
- "train two diffusion upsampler models [34, 43]: one to upsample images from  $64 \times 64$  to  $256 \times 256$  resolution" (Section 2.1 Decoder)
- "another to further upsample those to  $1024 \times 1024$  resolution." (Section 2.1 Decoder)
- Inference: In/Out Dimensions and Dynamics labeled 2D with fixed sizes inferred from the stated resolutions; Attention Static and State Constructed inferred from conditioning on provided images within the diffusion upsamplers.

### Task: Image variation generation
- "decoders conditioned on image representations can also produce variations of an image" (Abstract)
- "Given an image x, we can produce related images that share the same essential content but vary" (Section 3.1 Variations)
- "encode any given image x into a bipartite latent representation  $(z_i, x_T)$" (Section 3 Image Manipulations)
- Inference: Dimensions/Dynamics inferred from the image domain and fixed resolutions; Attention Static inferred from conditioning on the given image; State Constructed inferred from the explicit bipartite latent representation.

### Task: Image interpolation / blending
- "We can also interpolate between input images by inverting interpolations of their image embeddings." (Section 1 Introduction)
- "It is also possible to blend two images  $x_1$  and  $x_2$" (Section 3.2 Interpolations)
- "encode any given image x into a bipartite latent representation  $(z_i, x_T)$" (Section 3 Image Manipulations)
- Inference: Dimensions/Dynamics inferred from the image domain and fixed resolutions; Attention Static inferred from conditioning on the given images; State Constructed inferred from the explicit bipartite latent representation.

### Task: Text-guided image manipulation (text diffs)
- "joint embedding space of CLIP enables language-guided image manipulations in a zero-shot fashion." (Abstract)
- "To modify the image to reflect a new text description y" (Section 3.3 Text Diffs)
- "Text encoder context 256" (Table 3, Training Details)
- "encode any given image x into a bipartite latent representation  $(z_i, x_T)$" (Section 3 Image Manipulations)
- Inference: Input dimensions/dynamics inferred from image + text inputs with capped text context; Attention Static inferred from conditioning on the given image and text; State Constructed inferred from the explicit bipartite latent representation.
