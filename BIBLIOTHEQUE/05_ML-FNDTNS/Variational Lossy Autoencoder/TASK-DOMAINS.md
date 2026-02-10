# VARIATIONAL LOSSY AUTOENCODER (Not specified in the paper.)
Source: Variational Lossy Autoencoder.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lossy compression / decompression | 2D images (binary or natural images) | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | Lossy latent code z; decompressed images | 1D (t) (inferred); 2D (x, y) | Fixed |
| Density estimation | 2D images (binary or natural images) | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | Image likelihood / NLL | 0D | Fixed |
| Image generation (sampling) | Latent noise/code samples (epsilon, z) | 1D (t) (inferred); 3D (x, y, z) (inferred) | Fixed | Static (inferred) | Constructed (inferred) | 2D images (generated samples) | 2D (x, y) | Fixed |

## Summary
The paper explicitly covers lossy compression/decompression and density estimation, and it also demonstrates direct image generation from latent noise/code samples. The dominant observed data modality is fixed-size 2D images, while outputs include both 2D generated/decompressed images and 0D likelihood metrics (NLL). Based on decoder design and latent-variable usage, attention is Static (inferred) and state is Constructed (inferred) across tasks.

## Evidence
### Task: Lossy compression / decompression
- "First we are interested in whether VLAE can learn a lossy representation/compression of data by using the PixelCNN decoder to model local statistics." (Section 4.1 Lossy Compression)
- "we visualize original images x_data and one random \"decompression\" x_decompressed from VLAE" (Section 4.1 Lossy Compression)
- "the global structure of the image was encoded in the lossy code z and regenerated." (Section 4.1 Lossy Compression)
- "All datasets uniformly consist of 28x28 binary images, which allow us to use a unified architecture." (Experiments)
- Inference: Attention Dynamic is Static (inferred) because "the window of dependency ... is limited to a small local patch," indicating a fixed, design-time context. State Dynamic is Constructed (inferred) because the model encodes and uses a latent variable "lossy code z" to regenerate outputs.

### Task: Density estimation
- "achieving new state-of-the-art results on MNIST, OMNIGLOT and Caltech-101 Silhouettes density estimation tasks" (Abstract)
- "For evaluation, we use binary image datasets that are commonly used for density estimation tasks" (Experiments)
- "Next we investigate whether leveraging autoregressive models as latent distribution p(z) and as decoding distribution p(x|z) would improve density estimation performance." (Section 4.2 Density Estimation)
- "Reported marginal NLL is estimated using Importance Sampling with 4096 samples." (Experiments)
- Inference: Attention Dynamic is Static (inferred) because decoding uses a predefined local receptive field. State Dynamic is Constructed (inferred) because density modeling uses latent variables and autoregressive structure (e.g., "global structure in latent code and local statistics in PixelCNN").

### Task: Image generation (sampling)
- "(b) Samples from VLAE" (Figure 1 caption, Section 4.1)
- "Figure 4: CIFAR10: Generated samples for different models" (Appendix E)
- "For an autoregressive flow f, some continuous noise source epsilon is transformed into latent code z: z = f(epsilon)." (Section 3.2 Learned Prior with Autoregressive Flow)
- "A latent code of dimension 64 was used." (Appendix A)
- "Latent codes are represented by 16 feature maps of size 8x8" (Appendix B)
- Inference: In Dimension includes 1D (t) (inferred) and 3D (x, y, z) (inferred) from the two latent-code forms above; Attention Dynamic is Static (inferred) from fixed autoregressive decoder receptive fields; State Dynamic is Constructed (inferred) because generation proceeds through constructed latent variables.
