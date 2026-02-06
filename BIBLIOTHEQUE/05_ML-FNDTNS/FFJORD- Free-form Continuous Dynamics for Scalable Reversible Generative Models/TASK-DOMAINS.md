# FFJORD: FREE-FORM CONTINUOUS DYNAMICS FOR SCALABLE REVERSIBLE GENERATIVE MODELS (Not specified in the paper)
Source: FFJORD- Free-form Continuous Dynamics for Scalable Reversible Generative Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| density estimation | tabular datasets; image datasets; 2 dimensional data | 2D (x, y); 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | log-density / log-likelihood | 0D (inferred) | Fixed (inferred) |
| generation (image generation) | samples from a fixed base distribution | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | images (MNIST, CIFAR10 samples) | 2D (x, y) (inferred) | Fixed (inferred) |
| variational inference | data x (VAE inputs) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | flow parameter updates (low-rank update matrix; bias vector) | Not specified in the paper. | Not specified in the paper. |

## Summary
FFJORD is evaluated for density estimation on tabular and image data (including explicit 2D toy data) and for image generation via sampling from a base distribution. It is also used as a flow in VAEs for variational inference. The paper supports 2D data explicitly and fixed-size vector/scalar interfaces by inference, while attention and state dynamics are not described.

## Evidence
### Task: density estimation
- "We demonstrate our approach on high-dimensional density estimation, image generation, and variational inference." (Abstract)
- "We perform density estimation on five tabular datasets preprocessed as in Papamakarios et al. (2017) and two image datasets; MNIST and CIFAR10." (Section 4.2)
- "We first train on 2 dimensional data to visualize the model and the learned dynamics." (Section 4.1)
- "define a generative model for data  $\mathbf{x} \in \mathbb{R}^D$" (Section 2.2)
- "Given a datapoint x, we can compute both the point  $z_0$  which generates x, as well as  $\log p(x)$  under the model" (Section 2.2)
- Inference: Labeled tabular inputs as `1D (t)` and `Fixed` because the data are defined as $\mathbf{x} \in \mathbb{R}^D$ (Section 2.2). Labeled output as `0D` because it computes $\log p(x)$ (Section 2.2).

### Task: generation (image generation)
- "We demonstrate our approach on high-dimensional density estimation, image generation, and variational inference." (Abstract)
- "Reversible generative models use cheaply invertible neural networks to transform samples from a fixed base distribution." (Section 1 Introduction)
- "The generative process works by first sampling from a base distribution  $\mathbf{z}_0 \sim p_{z_0}(\mathbf{z}_0)$ ." (Section 2.2)
- "Samples and data from our image models. MNIST on left, CIFAR10 on right." (Figure 3 caption)
- Inference: Marked the input as `1D (t)` and `Fixed` from the fixed base distribution and the model operating on $\mathbb{R}^D$ (Section 1 Introduction, Section 2.2). Marked the outputs as `2D (x, y)` and `Fixed` because the task is image generation on MNIST/CIFAR10 (Section 4.2, Figure 3 caption).

### Task: variational inference
- "We demonstrate our approach on high-dimensional density estimation, image generation, and variational inference." (Abstract)
- "We compare FFJORD to other normalizing flows for use in variational inference." (Section 4.3)
- "In VAEs it is common for the encoder network to also output the parameters of the flow as a function of the input x." (Section 4.3)
