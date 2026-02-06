# Denoising Diffusion Probabilistic Models (2020)
Source: Denoising Diffusion Probabilistic Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image synthesis (unconditional generation) | Gaussian noise latent (x_T) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | images (samples) | 2D (x, y) (inferred) | Fixed (inferred) |
| Progressive lossy image compression | images (x_0) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | bits | 1D (t) (inferred) | Not specified in the paper. |
| Image interpolation (latent-space) | source images (x_0, x_0') | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | interpolated images | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper primarily targets unconditional image synthesis, and also demonstrates latent-space image interpolation plus a progressive lossy image compression scheme. The modalities are 2D image grids at fixed resolutions, while compression outputs a sequential bitstream; these Dimension values are inferred from the described datasets and coding procedure. Output dynamics for the compression bitstream are not specified, and attention/state dynamics are inferred as static/direct because the model operates on fixed-size image latents with Markov-chain transitions and no described runtime selection or external memory.

## Evidence
### Task: Image synthesis (unconditional generation)
- "We present high quality image synthesis results using diffusion probabilistic models," (Abstract)
- "On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN." (Abstract)
- "a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time." (Section 1 Introduction)
- "$p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$" (Section 2 Background)
- "$\mathbf{x}_1, \dots, \mathbf{x}_T$ are latents of the same dimensionality as the data $\mathbf{x}_0$" (Section 2 Background)
- Inference: Input/output dimensions and fixed dynamics are inferred from the stated shared dimensionality of x_0 and x_1:T and the fixed image datasets; attention and state are inferred as static/direct because the model is a fixed-size Markov chain over x_t without any described runtime selection or external memory.

### Task: Progressive lossy image compression
- "we conclude that diffusion models have an inductive bias that makes them excellent lossy compressors." (Section 4.3 Progressive coding)
- "introducing a progressive lossy code that mirrors the form of Eq. (5)" (Section 4.3 Progressive coding)
- "transmit a sample  $\mathbf{x} \sim q(\mathbf{x})$  using approximately  $D_{\mathrm{KL}}(q(\mathbf{x}) \parallel p(\mathbf{x}))$  bits on average" (Section 4.3 Progressive coding)
- "Algorithms 3 and 4 transmit  $\mathbf{x}_T, \ldots, \mathbf{x}_0$  in sequence" (Section 4.3 Progressive coding)
- Inference: The output is treated as a 1D sequential bitstream because transmission is described as bits sent in sequence over diffusion steps; output dynamics are not specified, and attention/state are inferred as static/direct for the same fixed-size Markov-chain reason as above.

### Task: Image interpolation (latent-space)
- "We can interpolate source images  $\mathbf{x}_0, \mathbf{x}_0' \sim q(\mathbf{x}_0)$  in latent space using q as a stochastic encoder," (Section 4.4 Interpolation)
- "decoding the linearly interpolated latent  $\bar{\mathbf{x}}_t = (1-\lambda)\mathbf{x}_0 + \lambda\mathbf{x}_0'$  into image space by the reverse process" (Section 4.4 Interpolation)
- "Figure 8: Interpolations of CelebA-HQ 256x256 images with 500 timesteps of diffusion." (Section 4.4 Interpolation)
- Inference: Dimensions and fixed dynamics are inferred from the same fixed-size image setting; attention and state are inferred as static/direct because the interpolation procedure applies the same fixed-size reverse process without runtime selection or external memory.
