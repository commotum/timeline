# High-Resolution Image Synthesis with Latent Diffusion Models (Not specified in the paper.)
Source: High-Resolution Image Synthesis with Latent Diffusion Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unconditional image synthesis | Normally distributed variable (noise) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) (inferred) | Fixed (inferred) |
| Class-conditional image synthesis | Class labels | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) (inferred) | Fixed (inferred) |
| Text-to-image synthesis | Language prompts (tokenized text) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) (inferred) | Capped (inferred) |
| Layout-to-image synthesis | Semantic layouts (bounding boxes) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) (inferred) | Not specified in the paper. |
| Semantic-map-to-image synthesis | Semantic maps | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) (inferred) | Capped (inferred) |
| Super-resolution | Low-resolution images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | High-resolution images | 2D (x, y) (inferred) | Capped (inferred) |
| Inpainting | Masked images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Inpainted images | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper covers unconditional and class-conditional image synthesis plus text-to-image, layout-to-image, semantic-map-to-image, super-resolution, and inpainting. Inputs are primarily images or semantic maps (2D), with text prompts (1D token sequences) and class labels (0D), and outputs are 2D images. Dynamics are fixed where explicit resolutions or sequence lengths are stated, while text-to-image and spatially conditioned tasks are described as scaling to larger outputs (capped), with attention/state inferred as static/direct from fixed conditioning via concatenation/cross-attention.

## Evidence
### Task: Unconditional image synthesis
- "unconditional image synthesis" (Introduction, contributions)
- "We train unconditional models of 256<sup>2</sup> images" (Sec. 4.2)
- "denoising a normally distributed variable" (Sec. 3.2)
- "two-dimensional structure of our learned latent space  $z = \mathcal{E}(x)$" (Sec. 3.1)
- Inference: Marked input/output dimensions as 2D and dynamics as Fixed based on the two-dimensional latent space and 256<sup>2</sup> image resolution; attention/state marked Static/Direct from fixed UNet conditioning (Sec. 3.3, Fig. 3).

### Task: Class-conditional image synthesis
- "class-conditional image synthesis" (Abstract)
- "class-conditional ImageNet [12], each with a resolution of  $256 	imes 256$" (Fig. 4)
- "mapping classes y to  $\zeta \in \mathbb{R}^{1 	imes 512}$" (Sec. E.2.1)
- Inference: Input labeled 0D and Fixed because classes map to a single embedding; output labeled 2D/Fixed from the stated  $256 	imes 256$  resolution; attention/state marked Static/Direct from fixed conditioning (Sec. 3.3, Fig. 3).

### Task: Text-to-image synthesis
- "text-to-image synthesis" (Abstract)
- "LDM conditioned on language prompts" (Sec. 4.3.1)
- "tokenized version of the input y" (Sec. E.2.1)
- "seq-length               | 77" (Table 17)
- "rendering images larger than the native  $256^2$  resolution" (Fig. 13)
- Inference: Input labeled 1D and Fixed from tokenized sequences with seq-length 77; output dynamics labeled Capped from larger-than-native  $256^2$  rendering; attention/state marked Static/Direct from fixed cross-attention conditioning (Sec. 3.3).

### Task: Layout-to-image synthesis
- "layout-to-image models" (Introduction, contributions)
- "synthesize images based on semantic layouts" (Sec. 4.3.1)
- "layout-to-image model discretizes the spatial locations of the bounding boxes" (Sec. E.2.1)
- "seq-length               | 92" (Table 17)
- Inference: Input labeled 2D from bounding-box spatial coordinates and Fixed from seq-length 92; output labeled 2D from synthesized images; attention/state marked Static/Direct from fixed cross-attention conditioning (Sec. 3.3).

### Task: Semantic-map-to-image synthesis
- "train models for semantic synthesis" (Sec. 4.3.2)
- "images of landscapes paired with semantic maps" (Sec. 4.3.2)
- "We train on an input resolution of  $256^2$" (Sec. 4.3.2)
- "generate images up to the megapixel regime" (Sec. 4.3.2)
- Inference: Input/output labeled 2D from semantic maps and images; dynamics labeled Capped from training at  $256^2$  and generating up to megapixel resolution; attention/state marked Static/Direct from fixed concatenation conditioning (Sec. 4.3.2, Fig. 3).

### Task: Super-resolution
- "super-resolution" (Abstract)
- "conditioning on low-resolution images" (Sec. 4.4)
- "ImageNet  $64 ightarrow 256$  super-resolution" (Fig. 10)
- "generate large images between 5122 and 1024<sup>2</sup>" (Sec. 4.3.2)
- Inference: Input/output labeled 2D from low- and high-resolution images; dynamics labeled Capped from  $64 ightarrow 256$  scaling and larger-image generation; attention/state marked Static/Direct from fixed concatenation conditioning (Sec. 4.4).

### Task: Inpainting
- "image inpainting" (Abstract)
- "Inpainting is the task of filling masked regions of an image with new content" (Sec. 4.5)
- "resolution  $256^2$  and  $512^2$" (Sec. 4.5)
- "generate large images between 5122 and 1024<sup>2</sup>" (Sec. 4.3.2)
- Inference: Input/output labeled 2D from masked images and reconstructed images; dynamics labeled Capped from multiple stated resolutions and larger-image generation; attention/state marked Static/Direct from fixed concatenation conditioning (Sec. 4.5).
