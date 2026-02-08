# Zero-Shot Text-to-Image Generation (Year not specified in the paper)
Source: Zero-Shot Text-to-Image Generation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text-to-image generation (zero-shot) | Captions (text tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Images | 2D (x, y) (inferred) | Fixed (inferred) |
| Zero-shot image-to-image translation (natural-language controlled) | Caption text tokens + partial image token grid (top 15 x 32) | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Translated images | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper covers two generation/manipulation tasks: zero-shot text-to-image generation and zero-shot image-to-image translation controllable by natural language. Inputs are 1D text for text-to-image, and combined 1D text plus 2D image context for image-to-image translation; outputs are 2D images in both cases. The OCR specifies a capped text interface (up to 256 tokens) and fixed image token geometry (32 x 32), supporting Capped input dynamics and Fixed output dynamics. The transformer uses predefined sparse masks over a fixed token stream, which supports Static attention and Direct state (both inferred).

## Evidence
### Task: Text-to-image generation (zero-shot)
- "We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data." (Abstract)
- "The resulting system achieves high quality image generation on the popular MS-COCO dataset zero-shot, without using any of the training labels." (Section 1, Introduction)
- "We concatenate up to 256 BPE-encoded text tokens with the  $32 \times 32 = 1024$  image tokens, and train an autoregressive transformer to model the joint distribution over the text and image tokens." (Section 2, Method)
- Inference: `1D (t)` and `Capped` input are inferred from text-token sequence input and the explicit 256-token cap. `2D (x, y)` and `Fixed` output are inferred from the fixed 32 x 32 image token grid and fixed-resolution image compression ("compress each  $256 \times 256$  RGB image into a  $32 \times 32$  grid of image tokens," Section 2, Method). `Static` attention is inferred from predefined masks ("the part for the image-to-image attention uses either a row, column, or convolutional attention mask," Section 2.2). `Direct` state is inferred because the model is described as autoregressively modeling token streams without an explicit constructed state mechanism.

### Task: Zero-shot image-to-image translation (natural-language controlled)
- "To a limited degree of reliability, we also find our model to be capable of zero-shot image-to-image translation controllable by natural language (Figure 2d)." (Section 3.3, Qualitative Findings)
- "When the model is given the caption \"the exact same cat on the top as a sketch at the bottom\" and the top  $15 \times 32$  part of the image token grid for a photo of a cat, it is able to draw a sketch of a similar looking cat on the bottom." (Section 3.3, Qualitative Findings)
- "This works with several other kinds of transformations, including image operations (e.g., changing the color of the image, converting it to grayscale, or flipping it upside-down) and style transfer..." (Section 3.3, Qualitative Findings)
- Inference: Input spans text plus image context, so `1D (t); 2D (x, y)` is inferred from the caption and partial image grid input. `Capped` input is inferred from the 256-token text cap (Section 2.2) and fixed image token geometry (`32 x 32`, Section 2). Output is inferred as `2D (x, y)` with `Fixed` dynamics from the same fixed image-token setup used by the model. `Static` attention and `Direct` state are inferred from the same fixed-mask autoregressive transformer design described in Section 2.2.
