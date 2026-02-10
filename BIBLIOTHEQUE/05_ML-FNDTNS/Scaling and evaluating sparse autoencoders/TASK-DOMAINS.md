# Scaling and evaluating sparse autoencoders (2024)
Source: Scaling and evaluating sparse autoencoders.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sparse activation reconstruction | Language-model residual stream activation vectors (token-wise) | 1D (t) (inferred) | Fixed | Static (inferred) | Constructed (inferred) | Sparse latent activations and reconstructed residual stream activations | 1D (t) (inferred) | Fixed |

## Summary
The paper focuses on a single core task: reconstructing language-model residual activations through a sparse autoencoder bottleneck. The covered modality is token-sequence-derived residual stream data, with a fixed experiment interface (64-token contexts) and fixed-size per-token activation vectors. Based on the architecture and setup, the task is best characterized as static-attention and constructed-state, with 1D token-order structure inferred from the explicit context-length and token-position analyses.

## Evidence
### Task: Sparse activation reconstruction
- "Sparse autoencoders provide a promising unsupervised approach for extracting interpretable features from a language model by reconstructing activations from a sparse bottleneck layer." (Abstract)
- "For an input vector  $x \in \mathbb{R}^d$  from the residual stream ..." and "$$\hat{x} = W_{\text{dec}}z + b_{\text{pre}}$$" (Section 2.2 Baseline: ReLU autoencoders)
- "We use a context length of 64 tokens for all experiments." (Section 2.1 Setup)
- Inference: `1D (t)` is inferred because the paper ties activations to token order and context length ("context length of 64 tokens"; Section 2.1 and token-position analysis in Section F.2). `Static` attention is inferred because the SAE consumes a predefined activation vector rather than selecting new observations at runtime (Sections 2.2-2.3). `Constructed` state is inferred because the sparse latent bottleneck `z` is a learned internal abstraction used to reconstruct `x` (Sections 2.2-2.3).
