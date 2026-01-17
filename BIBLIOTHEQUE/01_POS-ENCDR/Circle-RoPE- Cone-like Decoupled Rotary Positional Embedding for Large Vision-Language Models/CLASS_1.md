# CIRCLE-ROPE: CONE-LIKE DECOUPLED ROTARY POSI-TIONAL EMBEDDING FOR LARGE VISION-LANGUAGE MODELS (Not specified in the paper.)
Source: Circle-RoPE- Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models.md

## Core reasons
- The paper critiques existing RoPE variants for VLMs as introducing cross-modal positional biases when applied to text and image tokens.
- The main contribution is a new positional encoding scheme (Circle-RoPE) that modifies RoPE to decouple text and image positional relationships.

## Evidence extracts
- "However, when extended to vision-language models (VLMs), RoPE and its variants enforce relative positional dependencies separately within text and image tokens, introducing unintended cross-modal positional biases." (Abstract)
- "We propose a novel positional encoding method for VLMs, **Circle Rotary Position Embedding** (**Circle-RoPE**)." (Section 4 METHOD)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
