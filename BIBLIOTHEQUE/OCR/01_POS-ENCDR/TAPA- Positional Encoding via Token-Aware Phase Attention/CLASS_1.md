# Positional Encoding via Token-Aware Phase Attention (Not specified in the paper.)
Source: TAPA- Positional Encoding via Token-Aware Phase Attention.md

## Core reasons
- The paper critiques RoPE's inability to extrapolate to longer contexts, framing a limitation of existing positional encoding.
- The core contribution is a new positional encoding method (TAPA) that modifies attention with a learnable phase function.

## Evidence extracts
- "However, RoPE, as originally designed, is not able to extrapolate to context lengths that were not seen during pretraining (Sun et al., 2022), even with extensive continual pretraining at the extended lengths (Chen et al., 2023; Xiong et al., 2023)." (Section 1 Introduction)
- "We introduce Token-Aware Phase Attention (TAPA), a simple positional encoding framework that inserts a learnable phase function into the attention mechanism." (Section 1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
