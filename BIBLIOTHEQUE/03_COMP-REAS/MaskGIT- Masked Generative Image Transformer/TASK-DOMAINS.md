# MaskGIT: Masked Generative Image Transformer (Not specified in the paper)
Source: MaskGIT- Masked Generative Image Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class-conditional image synthesis | class labels (inferred); masked image tokens (all masked) | 0D (inferred); 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |
| class-conditional image editing | images; bounding boxes; class labels (inferred) | 2D (x, y) (inferred); 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |
| image inpainting | masked images; inpainting masks | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |
| image outpainting (extrapolation) | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |
| image reconstruction | masked image tokens | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
MaskGIT is used for image generation and editing tasks, including class-conditional image synthesis, class-conditional editing, inpainting, outpainting/extrapolation, and reconstruction from masked tokens. The inputs and outputs described are images (and, for class-conditional tasks, class labels) arranged as H×W grids, with experiments reported at fixed resolutions such as 256×256 and 512×512. Based on the bidirectional self-attention and iterative refinement described, the attention is treated as static over the full grid while the state is constructed across iterations.

## Evidence
### Task: class-conditional image synthesis
- "class-conditional image synthesis" (Section 4.2)
- "start from a blank canvas with all the tokens masked out" (Section 3.2)
- Inference: Treated class labels as inputs based on "class-conditional image synthesis" (Section 4.2). Treated image/tokens as 2D fixed grids because the paper describes "images  $x \in \mathbb{R}^{H \times W \times 3}$" and uses "cropped 256x256 images for all the experiments" (Sections 2.1, 4.1). Marked attention as Static from "self-attention allows the model to generate new tokens from generated tokens in all directions" and state as Constructed because it "refines the image iteratively conditioned on the previous generation" (Introduction, Abstract).

### Task: class-conditional image editing
- "class-conditional image editing task" (Section 4.3)
- "regenerates content specified inside a bounding box on the given class" (Section 4.3)
- Inference: Interpreted the "given class" as a class-label input. Treated image outputs as 2D fixed grids because the paper describes "images  $x \in \mathbb{R}^{H \times W \times 3}$" and uses "cropped 256x256 images for all the experiments" (Sections 2.1, 4.1). Marked attention as Static from "self-attention allows the model to generate new tokens from generated tokens in all directions" and state as Constructed because it "refines the image iteratively conditioned on the previous generation" (Introduction, Abstract).

### Task: image inpainting
- "Image inpainting or image completion" (Section 4.3)
- "tokenizing the masked image and interpreting the inpainting mask as the initial mask" (Section 4.3)
- Inference: Treated image inputs/outputs as 2D fixed grids because the paper describes "images  $x \in \mathbb{R}^{H \times W \times 3}$" and uses "cropped 256x256 images for all the experiments" (Sections 2.1, 4.1). Marked attention as Static from "self-attention allows the model to generate new tokens from generated tokens in all directions" and state as Constructed because it "refines the image iteratively conditioned on the previous generation" (Introduction, Abstract).

### Task: image outpainting (extrapolation)
- "Outpainting, or image extrapolation" (Section 4.3)
- "outpainting in different directions" (Figure 7)
- Inference: Treated image inputs/outputs as 2D fixed grids because the paper describes "images  $x \in \mathbb{R}^{H \times W \times 3}$" and uses "cropped 256x256 images for all the experiments" (Sections 2.1, 4.1). Marked attention as Static from "self-attention allows the model to generate new tokens from generated tokens in all directions" and state as Constructed because it "refines the image iteratively conditioned on the previous generation" (Introduction, Abstract).

### Task: image reconstruction
- "outputs reconstructed images" (Figure 9)
- "iterative decoding algorithm to reconstruct images" (Appendix A)
- Inference: Treated image/token inputs and outputs as 2D fixed grids because the paper describes "images  $x \in \mathbb{R}^{H \times W \times 3}$" and uses "cropped 256x256 images for all the experiments" (Sections 2.1, 4.1). Marked attention as Static from "self-attention allows the model to generate new tokens from generated tokens in all directions" and state as Constructed because it "refines the image iteratively conditioned on the previous generation" (Introduction, Abstract).

## CSV Output (required)
