# One-Step Diffusion Distillation via Deep Equilibrium Models (Generative Equilibrium Transformer, GET) (Not specified in the paper)
Source: One-Step Diffusion Distillation via Deep Equilibrium Models (Generative Equilibrium Transformer, GET).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image generation (unconditional) | Gaussian noise image | 2D (x, y) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Image | 2D (x, y) | Fixed (inferred) |
| Image generation (class-conditional) | Gaussian noise image; class label | 2D (x, y); 0D | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Image | 2D (x, y) | Fixed (inferred) |

## Summary
The paper focuses on single-step image generation, evaluating both unconditional and class-conditional settings that map Gaussian noise (optionally with class labels) to images. Inputs and outputs are specified as $H \times W \times C$ tensors, so the tasks operate over 2D (x, y) image grids and produce 2D images. Attention, state, and fixed-size dynamics are inferred from the transformer-based DEQ architecture that solves for a fixed-point latent representation.

## Evidence
### Task: Image generation (unconditional)
- "We evaluate the effectiveness of our proposed Generative Equilibrium Transformer (GET) in offline distillation of diffusion models through a series of experiments on single-step class-conditional and unconditional image generation." (Section 4 Experiments)
- "GET first converts an input noise  $\mathbf{e} \in \mathbb{R}^{H \times W \times C}$  into a sequence of 2D patches  $\mathbf{p} \in \mathbb{R}^{N \times (P^2 \cdot C)}$" (Section 3, Noise Embedding)
- "The resulting patches  $\bar{\mathbf{p}}$  are rearranged back to the resolution of the input noise e to produce the image sample  $\hat{\mathbf{x}} \in \mathbb{R}^{H \times W \times C}$ ." (Section 3, Image Decoder)
- "The EquilibriumT, which is the equilibrium layer, solves for the fixed point" (Section 3, InjectionT & EquilibriumT)
- Inference: Marked In/Out Dynamics as Fixed (inferred) because the input and output are specified as fixed-resolution tensors $\mathbb{R}^{H \times W \times C}$ with patch embedding/decoding. Marked Attention Dynamic as Static (inferred) because the architecture uses standard transformer attention over a fixed token sequence. Marked State Dynamic as Constructed (inferred) because the model solves for a fixed-point latent state before decoding to an image.

### Task: Image generation (class-conditional)
- "We evaluate the effectiveness of our proposed Generative Equilibrium Transformer (GET) in offline distillation of diffusion models through a series of experiments on single-step class-conditional and unconditional image generation." (Section 4 Experiments)
- "Generative Equilibrium Transformer (GET) directly maps Gaussian noises e and optional class labels e to images  $\tilde{e}$ ." (Section 3, GET)
- "To train a class-conditional GET, we also use class labels  $\mathbf{y}$  in addition to noise/image pairs:" (Section 4, Offline Distillation)
- "The resulting patches  $\bar{\mathbf{p}}$  are rearranged back to the resolution of the input noise e to produce the image sample  $\hat{\mathbf{x}} \in \mathbb{R}^{H \times W \times C}$ ." (Section 3, Image Decoder)
- "The EquilibriumT, which is the equilibrium layer, solves for the fixed point" (Section 3, InjectionT & EquilibriumT)
- Inference: Marked In/Out Dynamics as Fixed (inferred) because the input and output are specified as fixed-resolution tensors $\mathbb{R}^{H \times W \times C}$ with patch embedding/decoding, and the class label input is a fixed-size token. Marked Attention Dynamic as Static (inferred) because the architecture uses standard transformer attention over a fixed token sequence. Marked State Dynamic as Constructed (inferred) because the model solves for a fixed-point latent state before decoding to an image.
