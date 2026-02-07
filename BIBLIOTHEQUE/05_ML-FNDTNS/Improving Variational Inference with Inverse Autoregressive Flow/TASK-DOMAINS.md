# Improved Variational Inference with Inverse Autoregressive Flow (Not specified in the paper.)
Source: Improving Variational Inference with Inverse Autoregressive Flow.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| posterior inference (variational inference) | observed variables x (datapoint) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | approximate posterior over latent variables z (samples/parameters) | 1D (t); 3D (x, y, z) | Not specified in the paper. |
| density estimation (log-likelihood) | images (MNIST/CIFAR-10) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | log-likelihood / bits per dimension | 0D (inferred) | Not specified in the paper. |
| generation (image synthesis) | latent variables z / noise samples | 1D (t); 3D (x, y, z) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | images | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper centers on variational posterior inference for latent-variable models and evaluates image generative modeling on MNIST and CIFAR-10 using log-likelihood metrics and sampling-based synthesis. The covered inputs/outputs include latent vectors or 3D feature-map tensors and 2D image grids, while likelihood outputs are scalar scores. Dynamics and attention/state behaviors are largely not specified beyond these modalities.

## Evidence
### Task: posterior inference (variational inference)
- "The framework of normalizing flows provides a general strategy for flexible variational inference of posteriors over latent variables." (Abstract)
- "A solution is to introduce  $q(\mathbf{z}|\mathbf{x})$ , a parametric *inference model* defined over the latent variables" (Section 2)
- "x: a datapoint, and optionally other conditioning information" (Algorithm 1)
- "z: a random sample from q(z|x), the approximate posterior distribution" (Algorithm 1)
- "each stochastic variable is a three-dimensional tensor (a stack of featuremaps)" (Introduction)
- "$\mathbf{y} = \{y_i\}_{i=1}^D$" (Section 3)

### Task: density estimation (log-likelihood)
- "attained log-likelihood on natural images" (Abstract)
- "Table 1: Generative modeling results on the dynamically sampled binarized MNIST version used in previous publications" (Section 6.1)
- "Our architecture with IAF achieves **3.11 bits per dimension**" (Section 6.2)
- Inference: Input dimension set to 2D (x, y) because the task is on MNIST/CIFAR-10 images (Section 6.1; Section 6.2); output dimension set to 0D because log-likelihood/bits per dimension are scalar scores (Abstract; Section 6.2).

### Task: generation (image synthesis)
- "allowing significantly faster synthesis." (Abstract)
- "Sampling took about **0.05 seconds/image** with the ResNet VAE model" (Section 6.2)
- "$p(\mathbf{x}|\mathbf{z}_{1:L})p(\mathbf{z}_{1:L})$" (Section C ResNet VAE)
- Inference: Output dimension set to 2D (x, y) because the generated outputs are images (Section 6.2).
