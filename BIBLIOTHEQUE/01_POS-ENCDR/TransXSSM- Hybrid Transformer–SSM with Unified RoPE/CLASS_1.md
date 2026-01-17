# TransXSSM: A Hybrid Transformer-State Space Model with Unified Rotary Position Embedding (Not specified in the paper.)
Source: TransXSSM- Hybrid Transformer–SSM with Unified RoPE.md

## Core reasons
- The paper identifies a positional encoding mismatch between Transformers and SSMs and frames it as a key limitation in hybrid models.
- The main contribution is a new unified rotary positional encoding that modifies how positions are encoded across attention and state-space layers.

## Evidence extracts
- "Transformers rely on explicit Rotary Position Embeddings (RoPE), while SSMs leverage implicit positional representations via convolutions. This divergence often precipitates discontinuities and suboptimal performance. To address this impediment, we propose a unified rotary position embedding (Unified RoPE) methodology" (Abstract)
- "We propose a unified rotary position encoding that applies the same rotational embedding to both self-attention (Transformer) and state-space (SSM) components." (Section 2.2)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
