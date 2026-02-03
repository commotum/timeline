# MemoryLLM: Plug-n-Play Interpretable Feed-Forward Memory for Transformers (2026)
Source: MemoryLLM- Plug-n-Play Interpretable Feed-Forward Memory for Transformers.md

## Core reasons
- Proposes a modified transformer computation that decouples FFNs from self-attention so FFNs act as context-free memory, changing how inference is performed.
- Frames FFNs as a token-indexed neural memory mechanism trained in isolation, which is a computation mechanism proposal rather than positional encoding or dataset work.

## Evidence extracts
- "MemoryLLM decouples FFNs across all transformer blocks completely from self-attention modules and trains them in isolation" (Section 1 Introduction, Figure 1 caption)
- "FFNs are trained in isolation using context-free token embeddings, enabling their interpretation as neural key-value memory" (Section 5 Conclusion)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
