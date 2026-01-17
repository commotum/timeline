# YaRN: Efficient Context Window Extension of Large Language Models (Not specified in the paper.)
Source: YaRN- Efficient Context Window Extension of Large Language Models.md

## Core reasons
- The paper critiques RoPE-based models for failing to generalize beyond trained context lengths and targets positional encoding limitations directly.
- The core contribution is a modified RoPE-based positional encoding method (YaRN) that combines interpolation and attention scaling to extend context windows.

## Evidence extracts
- "Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models" (Abstract)
- "By the \"YaRN method\", we refer to a combination of the attention scaling in Eq. 21 and the \"NTK-by-parts\" interpolation introduced in Section 3.2." (Section 3.4 YaRN)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
