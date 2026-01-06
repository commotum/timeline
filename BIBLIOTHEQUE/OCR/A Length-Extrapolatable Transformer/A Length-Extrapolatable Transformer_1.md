# A Length-Extrapolatable Transformer (Not specified in the paper.)
Source: A Length-Extrapolatable Transformer.md

## Core reasons
- The paper critiques existing positional encoding methods for poor length extrapolation and identifies limitations in RoPE and Alibi.
- The main contribution is a new relative positional encoding (XPOS) and related attention changes to improve length extrapolation in Transformers.

## Evidence extracts
- "However, it can't deal with sequences with exceed length. Alibi (Press et al., 2021) mitigates the extrapolation problem but sacrifices the general performance." (1 Introduction)
- "Considering all the properties above, we propose Extrapolatable Position Embedding (XPOS), which is a universal-good design for Transformers." (1 Introduction)
- "Specifically, we introduce a relative position embedding to explicitly maximize attention resolution." (Abstract)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
