# Fixed Point Diffusion Models (Not specified in the paper.)
Source: Fixed Point Diffusion Models (FPDM).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image generation | noisy latent images (from noise distribution $q(X_T)$); timestep t; optional class labels | 2D (x, y); 0D | Fixed | Static (inferred) | Direct (inferred) | images (samples from target distribution $q(X_0)$) | 2D (x, y) | Fixed |

## Summary
FPDM is a diffusion-based image generation model that denoises latent images over timesteps to produce samples from a target image distribution. The paper operates in latent image space at fixed 256 resolution, with class-conditional ImageNet generation and unconditional generation on other datasets. Attention and state dynamics are not explicitly defined; given the transformer denoiser and per-timestep mapping, they are treated as Static and Direct (inferred).

## Evidence
### Task: image generation
- "We introduce the Fixed Point Diffusion Model (FPDM), a novel approach to image generation" (Abstract)
- "The generative process then begins with a sample from the noise distribution  $q(X_T)$  and denoises it over a series of steps" (Sec. 3.3)
- "to obtain a sample from the target distribution  $q(X_0)$ ." (Sec. 3.3)
- "Finally, note that our denoising network operates in latent space rather than pixel space." (Sec. 3.2)
- "we apply a Variational Autoencoder [28, 40] to encode the input image into latent space" (Sec. 3.2)
- "a implicit timestep-conditioned fixed-point layer  $f_{\text{fp}}: X \times X \times T \to X$ ." (Sec. 3.2)
- "All experiments are performed at resolution 256." (Sec. 4.1)
- "The ImageNet experiments are class-conditional, whereas those on other datasets are unconditional." (Sec. 4.1)
- "recently, a vision transformer architecture [13, 50]." (Introduction)
- "The output  $x_{\mathrm{post}}^{(t)}$  is used to compute the loss (during training) or the input  $x_{\mathrm{input}}^{(t-1)}$  to the next timestep (during sampling)." (Sec. 3.2)
- Inference: Attention Dynamic marked Static (inferred) because the model uses a vision transformer architecture and no dynamic input selection is described (see quote above). State Dynamic marked Direct (inferred) because outputs feed directly to loss or the next timestep rather than persistent state (see quote above).

## CSV Output (required)
CSV written to "/home/jake/Developer/timeline/BIBLIOTHEQUE/03_COMP-REAS/Fixed Point Diffusion Models (FPDM)/.TASK-DOMAINS.csv.tmp.3e6bfa9178234495a5c441c31822917c"
