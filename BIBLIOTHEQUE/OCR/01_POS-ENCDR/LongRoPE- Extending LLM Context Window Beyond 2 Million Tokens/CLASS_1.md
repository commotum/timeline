# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (Not specified in the paper.)
Source: LongRoPE- Extending LLM Context Window Beyond 2 Million Tokens.md

## Core reasons
- The paper critiques limitations of existing positional interpolation for RoPE and frames the issue as positional embedding non-uniformity.
- The main contribution is a new positional interpolation scheme (LongRoPE) that modifies RoPE rescale factors to extend context length while keeping the architecture unchanged.

## Evidence extracts
- "However, positional embedding exhibits *complex non-uniform information entropy* in the Transformer architecture. Such subtle non-uniformity is not effectively leveraged by existing approaches, leading to information loss and hence limiting the context window size." (Section 1. Introduction)
- "Models extended via LongRoPE retain the original architecture with minor modifications to the positional embedding, and can reuse most pre-existing optimizations." (Abstract)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
