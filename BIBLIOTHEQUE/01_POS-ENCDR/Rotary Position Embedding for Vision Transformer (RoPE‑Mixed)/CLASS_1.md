# Rotary Position Embedding for Vision Transformer (Not specified in the paper.)
Source: Rotary Position Embedding for Vision Transformer (RoPE‑Mixed).md

## Core reasons
- The paper's main contribution is to improve positional encoding for vision transformers by applying RoPE and proposing a new 2D variant (RoPE-Mixed).
- It critiques existing positional embeddings (APE/RPB) for poor resolution-change handling, motivating a positional encoding modification.

## Evidence extracts
- "This paper aims to improve position embedding for vision transformers by applying an extended Rotary Position Embedding (RoPE) [29]." (Section 1 Introduction)
- "Although both position embeddings are effective for the transformer on fixed-resolution settings, they struggle with resolution changes, requiring flexibility and extrapolation in position embeddings." (Section 1 Introduction)
- "To cope with the diagonal direction of RoPE, we propose to use mixed axis frequencies for 2D RoPE, named RoPE-Mixed." (Section 1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
