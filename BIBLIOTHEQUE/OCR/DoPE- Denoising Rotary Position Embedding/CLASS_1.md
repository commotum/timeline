# DoPE: Denoising Rotary Position Embedding (Not specified in the paper.)
Source: DoPE- Denoising Rotary Position Embedding.md

## Core reasons
- The paper critiques RoPE's limitations on length extrapolation and attention behavior in Transformers.
- The core contribution is a new positional encoding modification (DoPE) that denoises RoPE via truncated matrix entropy and Gaussian reparameterization.

## Evidence extracts
- "Rotary Position Embedding (RoPE) in Transformer models has inherent limits that weaken length extrapolation." (Abstract)
- "We presented Denoising Positional Encoding (DoPE), a parameter-free approach that mitigates low-rank artifacts in Rotary Position Embedding through truncated matrix entropy analysis." (Section 6 Conclusion)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
