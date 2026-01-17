# CRoPE: Efficient Parametrization of Rotary Positional Embedding (Not specified in the paper.)
Source: CRoPE- Efficient Parametrization of Rotary Positional Embedding.md

## Core reasons
- Proposes a new parametrization of rotary positional embeddings (CRoPE) that changes how positional encoding is implemented in Q/K/V projections to reduce parameters.
- Explicitly critiques limitations in existing positional embedding schemes, including RoPE's redundancy in embedding space.

## Evidence extracts
- "Rotary positional embedding has become the state-of-the-art approach to encode position information in transformer-based models. While it is often succinctly expressed in complex linear algebra, we note that the actual implementation of Q/K/V-projections is not equivalent to a complex linear transformation. We argue that complex linear transformation is a more natural parametrization and saves near 50% parameters within the attention block." (Abstract)
- "However, positional embedding schemes have never been perfect[8, 9]. Early absolute embeddings made it hard for models to disentangle position from semantic content[1, 10]. Relative embeddings mitigated this but required extra parameters[11, 12, 13]. Rotary positional embedding (RoPE) removes the explicit parameterization, yet implicitly still reserves half of the embedding space for positional information[14]." (1 Introduction)
- "we have shown rewriting RoPE in complex forms naturally leads to the parametrization of CRoPE where Q/K/V-projections are implemented as complex linear transformation, saving nearly half the parameters in the attention layers." (7 Conclusion)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
