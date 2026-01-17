# Selective Rotary Position Embedding (Not specified in the paper.)
Source: Selective Rotary Position Embedding.md

## Core reasons
- Proposes a new rotary positional embedding mechanism (Selective RoPE) that generalizes RoPE for transformers.
- Identifies a limitation of prior gating/linear attention approaches: they decay norms but lack rotation-based positional encoding.

## Evidence extracts
- "we introduce Selective RoPE, an input-dependent rotary embedding mechanism, that generalizes RoPE, and enables rotation in arbitrary angles for both linear and softmax transformers." (Abstract)
- "These mechanisms largely operate by modulating norms of key-value associations (i.e., how quickly they decay), but do not directly provide the complementary capability of rotating query-key representations to encode relative position." (1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
