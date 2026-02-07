# GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models (2021)
Source: GLIDE- Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text-conditional image generation | Text prompts (captions) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Images | 2D (x, y) (inferred) | Fixed (inferred) |
| unconditional image generation | Empty text sequence | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Images | 2D (x, y) (inferred) | Fixed (inferred) |
| text-conditional image inpainting | Partially observed image + mask + text prompt | 2D (x, y); 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Inpainted image | 2D (x, y) (inferred) | Fixed (inferred) |
| text-conditional sketch-guided image editing (SDEdit) | Sketch image + text caption | 2D (x, y); 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Edited image | 2D (x, y) (inferred) | Fixed (inferred) |
| text-conditional image upsampling (super-resolution) | Low-resolution image + text prompt | 2D (x, y); 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Higher-resolution image | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
GLIDE covers text-conditional image generation and editing tasks (inpainting and sketch-guided SDEdit), and it also supports unconditional image generation plus a text-conditioned upsampling model from low- to higher-resolution images. Inputs include text prompts or empty text sequences along with images, masks, and sketches, producing 2D images. Dimensions and dynamics are inferred as 1D text and fixed-size 2D images, with attention and state dynamics inferred from the fixed token attention context and diffusion Markov-chain sampling process.

## Evidence
### Task: text-conditional image generation
- "We explore diffusion models for the problem of text-conditional image synthesis" (Abstract)
- "First, we train a 3.5 billion parameter diffusion model that uses a text encoder to condition on natural language descriptions." (Introduction)
- Inference: In/Out dimensions and fixed dynamics inferred from "text-conditional diffusion model at  $64 \times 64$  resolution" (Section 4) and "encode it into a sequence of K tokens" (Section 4.1). Attention static inferred from "concatenated to the attention context at each layer." (Section 4.1). State constructed inferred from "we produce a Markov chain of latent variables  $x_1, ..., x_T$" (Section 2.1).

### Task: unconditional image generation
- "we fine-tuned our base model to support unconditional image generation." (Section 4.2)
- "text token sequences are replaced with the empty sequence." (Section 4.2)
- Inference: In/Out dimensions and fixed dynamics inferred from "text-conditional diffusion model at  $64 \times 64$  resolution" (Section 4) and "encode it into a sequence of K tokens" (Section 4.1). Attention static inferred from "concatenated to the attention context at each layer." (Section 4.1). State constructed inferred from "we produce a Markov chain of latent variables  $x_1, ..., x_T$" (Section 2.1).

### Task: text-conditional image inpainting
- "fine-tuned our model to perform image inpainting, enabling powerful text-driven image editing." (Abstract)
- "random regions of training examples are erased, and the remaining portions are fed into the model along with a mask channel" (Section 4.3)
- Inference: In/Out dimensions and fixed dynamics inferred from "text-conditional diffusion model at  $64 \times 64$  resolution" (Section 4) and "encode it into a sequence of K tokens" (Section 4.1). Attention static inferred from "concatenated to the attention context at each layer." (Section 4.1). State constructed inferred from "we produce a Markov chain of latent variables  $x_1, ..., x_T$" (Section 2.1).

### Task: text-conditional sketch-guided image editing (SDEdit)
- "Examples of text-conditional SDEdit (Meng et al., 2021) with GLIDE, where the user combines a sketch with a text caption" (Figure 4 caption)
- "our model is capable of turning sketches into realistic image edits." (Section 5.1)
- Inference: In/Out dimensions and fixed dynamics inferred from "text-conditional diffusion model at  $64 \times 64$  resolution" (Section 4) and "encode it into a sequence of K tokens" (Section 4.1). Attention static inferred from "concatenated to the attention context at each layer." (Section 4.1). State constructed inferred from "we produce a Markov chain of latent variables  $x_1, ..., x_T$" (Section 2.1).

### Task: text-conditional image upsampling (super-resolution)
- "another 1.5 billion parameter text-conditional upsampling diffusion model to increase the resolution to  $256 \times 256$ ." (Section 4)
- "For the upsampling model, we always provide the full low-resolution image" (Section 4.3)
- Inference: In/Out dimensions and fixed dynamics inferred from "increase the resolution to  $256 \times 256$" (Section 4) and "encode it into a sequence of K tokens" (Section 4.1). Attention static inferred from "concatenated to the attention context at each layer." (Section 4.1). State constructed inferred from "we produce a Markov chain of latent variables  $x_1, ..., x_T$" (Section 2.1).
