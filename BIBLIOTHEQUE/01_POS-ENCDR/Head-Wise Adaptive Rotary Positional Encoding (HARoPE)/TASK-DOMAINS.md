# HEAD-WISE ADAPTIVE ROTARY POSITIONAL ENCODING FOR FINE-GRAINED IMAGE GENERATION (Not specified in the paper.)
Source: Head-Wise Adaptive Rotary Positional Encoding (HARoPE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (inferred) | images | 2D (x, y) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Fixed (inferred) |
| generation (class-conditional image) | class labels (inferred) | 0D (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | images | 2D (x, y) | Capped (inferred) |
| generation (text-to-image) | text prompts | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | images | 2D (x, y) | Capped (inferred) |

## Summary
The paper evaluates image understanding (classification) plus two image generation tasks: class-conditional ImageNet generation and text-to-image generation. Inputs span 2D images and 1D text prompts, while outputs are 0D class labels (inferred) or 2D images; image resolutions are reported at specific sizes, supporting capped dynamics where inferred. Attention and state dynamics are not explicitly specified.

## Evidence
### Task: classification (inferred)
- "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS)
- "Image understanding experiments use ImageNet at  $224 \times 224$  with standard resize and center-crop." (4.1 EXPERIMENTAL SETUPS - Dataset)
- "For image understanding, we report Top-1 accuracy." (4.1 EXPERIMENTAL SETUPS - Metrics)
- "Models are trained on the standard ImageNet-1k resolution of  $224 \times 224$  and tested at progressively larger resolutions." (4.3 ABLATION STUDY - Extrapolation)
- Inference: Task labeled as classification and outputs as class labels (0D, Fixed) inferred from use of Top-1 accuracy on ImageNet; input dynamics labeled Capped based on training at 224×224 and testing at larger resolutions.

### Task: generation (class-conditional image)
- "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS)
- "For ImageNet generation, we encode images using Stable Diffusion's VAE into  $z \in \mathbb{R}^{H/8 \times W/8 \times 4}$  with  $H \in \{128, 256, 512\}$ ." (4.1 EXPERIMENTAL SETUPS - Dataset)
- Inference: Input class labels (0D, Fixed) inferred from the "class-conditional image generation" task name; output dynamics labeled Capped based on the stated set of image resolutions.

### Task: generation (text-to-image)
- "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (4 EXPERIMENTS)
- "Text-to-image experiments with the FLUX model use the BLIP30-60k instruction-tuning set of 60k prompt-image pairs." (4.1 EXPERIMENTAL SETUPS - Dataset)
- "text-to-image generation at a high resolution of  $1024 \times 1024$" (4.3 ABLATION STUDY - Different Image Resolution)
- Inference: Input dimension labeled 1D (t) from the use of text prompts; output dynamics labeled Capped from the stated 1024×1024 resolution.
