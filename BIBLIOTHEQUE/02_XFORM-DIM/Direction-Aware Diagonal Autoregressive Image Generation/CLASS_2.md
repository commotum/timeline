# Direction-Aware Diagonal Autoregressive Image Generation (Not specified in the paper.)
Source: Direction-Aware Diagonal Autoregressive Image Generation.md

## Core reasons
- The paper targets image token sequences with two-dimensional spatial coordinates, motivating a transformer adaptation for 2D image generation rather than 1D text.
- The core method rearranges image tokens with a diagonal scanning order to improve autoregressive image generation, which is a structural adaptation for the 2D image domain.

## Evidence extracts
- "Unlike text sequences that inherently follow a unidirectional left-to-right ordering, discrete image token sequences produced by visual tokenizers maintain two-dimensional spatial coordinates." (Section 1. Introduction)
- "To address this issue, this paper proposes Direction-Aware Diagonal Autoregressive Image Generation (DAR) method, which generates image tokens following a diagonal scanning order." (Abstract)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
