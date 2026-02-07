# Image Transformer (2018)
Source: Image Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image generation (unconditional) | image pixels (previously generated) | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | image pixels | 2D (x, y) | Fixed |
| image generation (class-conditional) | image class embeddings; image pixels (previously generated) | 0D; 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | image pixels | 2D (x, y) | Fixed |
| image completion (conditional generation) | partial image (first half) | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | completed image pixels (second half / full image) | 2D (x, y) | Fixed |
| image super-resolution (4x) | low-resolution images (8x8) | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | high-resolution images (32x32) | 2D (x, y) | Fixed |

## Summary
The paper evaluates multiple image-generation tasks: unconditional generation, class-conditional generation, image completion, and 4x super-resolution. Inputs and outputs are 2D image grids (with class labels as additional 0D conditioning for class-conditional generation), and experiments use fixed-size images such as 32x32 outputs and 8x8 inputs for super-resolution. Attention is implemented with fixed local neighborhoods (Static, inferred), and the tasks are modeled autoregressively without additional constructed state (Direct, inferred).

## Evidence
### Task: image generation (unconditional)
- "Our unconditioned and class-conditioned image generation models both use 1D local attention" (Section 5.1)
- "We trained only unconditional generative models on ImageNet" (Section 5.1)
- "$\log p(x) = \sum_{t=1}^{h \cdot w \cdot 3} \log p(x_t \mid x_{< t})$" (Section 3.4)
- "produce  $32\times 32$  pixel images with 3072 positions" (Section 3.3)
- Inference: Attention Dynamic = Static (inferred) because attention is restricted to fixed local neighborhoods ("restricting the positions in the memory matrix M to a local neighborhood around the query position", Section 3.3). State Dynamic = Direct (inferred) because prediction conditions on previously generated pixels ("$\log p(x) = \sum_{t=1}^{h \cdot w \cdot 3} \log p(x_t \mid x_{< t})$", Section 3.4).

### Task: image generation (class-conditional)
- "In image-class conditional generation we condition on an embedding of one of a small number of image classes." (Introduction)
- "We represent the image classes as learned d-dimensional embeddings per class" (Section 5.2)
- "We trained the class-conditioned Image Transformer on CIFAR-10" (Section 5.2)
- "produce  $32\times 32$  pixel images with 3072 positions" (Section 3.3)
- Inference: Attention Dynamic = Static (inferred) because attention is restricted to fixed local neighborhoods ("restricting the positions in the memory matrix M to a local neighborhood around the query position", Section 3.3). State Dynamic = Direct (inferred) because prediction conditions on previously generated pixels ("$\log p(x) = \sum_{t=1}^{h \cdot w \cdot 3} \log p(x_t \mid x_{< t})$", Section 3.4).

### Task: image completion (conditional generation)
- "image completions by a conditional CIFAR-10 model" (Table 1)
- "image completions from our best conditional generation model, where we sample the second half." (Table 2)
- "produce  $32\times 32$  pixel images with 3072 positions" (Section 3.3)
- Inference: Attention Dynamic = Static (inferred) because attention is restricted to fixed local neighborhoods ("restricting the positions in the memory matrix M to a local neighborhood around the query position", Section 3.3). State Dynamic = Direct (inferred) because prediction conditions on previously generated pixels ("$\log p(x) = \sum_{t=1}^{h \cdot w \cdot 3} \log p(x_t \mid x_{< t})$", Section 3.4).

### Task: image super-resolution (4x)
- "We also present results on image super-resolution with a large magnification ratio (4x)." (Abstract)
- "Super-resolution is the process of recovering a high resolution image from a low resolution image while generating realistic and plausible details." (Section 5.3)
- "we enlarge an  $8\times 8$  pixel image four-fold to  $32\times 32$" (Section 5.3)
- "we resized the image to  $8\times 8$  pixels for the input and  $32\times 32$  pixels for the label" (Section 5.3)
- Inference: Attention Dynamic = Static (inferred) because attention is restricted to fixed local neighborhoods ("restricting the positions in the memory matrix M to a local neighborhood around the query position", Section 3.3). State Dynamic = Direct (inferred) because prediction conditions on previously generated pixels ("$\log p(x) = \sum_{t=1}^{h \cdot w \cdot 3} \log p(x_t \mid x_{< t})$", Section 3.4).
