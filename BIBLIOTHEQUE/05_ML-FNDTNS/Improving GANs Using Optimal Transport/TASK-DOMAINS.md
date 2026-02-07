# Improving GANs Using Optimal Transport (Not specified in the paper.)
Source: Improving GANs Using Optimal Transport.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image generation | random noise vectors (latent codes) | 1D (t) | Fixed | Not specified in the paper. | Not specified in the paper. | images | 2D (x, y) | Fixed |
| conditional image generation | side information s (text descriptions or labels) | 1D (t); 0D | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | images | 2D (x, y) | Not specified in the paper. |
| 2D point generation (mixture of Gaussians) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | 2D points (samples from a 2D mixture of Gaussians) | 2D (x, y) | Fixed |

## Summary
The paper focuses on generative modeling, demonstrating OT-GAN for unconditional image generation, conditional image generation from side information (text descriptions or labels), and a toy 2D Gaussian mixture generation task. Inputs include fixed-length latent noise vectors and side information, while outputs are 2D images or 2D point samples. Where specified, input/output sizes are fixed, and the paper does not specify attention or state dynamics.

## Evidence
### Task: image generation
- "A generator g and a discriminator d play a zero-sum game where the generator maps noise  $\mathbf{z}$  to simulated images  $\mathbf{y} = g(\mathbf{z})$  and where the discriminator tries to distinguish the simulated images  $\mathbf{y}$  from images  $\mathbf{x}$  drawn from the distribution of training data p." (Section 2)
- "The generator maps latent codes sampled from a 100 dimensional uniform distribution between -1 and 1 to  $32 \times 32$  color images." (Appendix B)

### Task: conditional image generation
- "Our algorithm for training generative models can be generalized to include conditional generation of images given some side information s, such as a text-description of the image or a label." (Section 4)
- "To further demonstrate the effectiveness of the proposed method on conditional image synthesis, we compare OT-GAN with state-of-the-art models on text-to-image generation" (Section 5.4)

### Task: 2D point generation (mixture of Gaussians)
- "We train generative models using different types of GAN on a 2D mixture of 8 Gaussians, with means arranged on a circle." (Section 5.1)
- "The goal for the generator is to recover all 8 modes." (Section 5.1)
