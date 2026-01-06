# MINIGPT-4: ENHANCING VISION-LANGUAGE UNDERSTANDING WITH ADVANCED LARGE LANGUAGE MODELS (Not specified in the paper.)
Source: MiniGPT-4.md

## Core reasons
- The paper's core contribution is a vision-language model that aligns a visual encoder with a large language model to handle image inputs, i.e., adapting Transformer-based language modeling to a higher-dimensional visual domain.
- The architecture explicitly combines a pretrained vision encoder (ViT + Q-Former) with an advanced LLM via a projection layer, emphasizing multimodal modeling rather than positional encoding changes.

## Evidence extracts
- "we present MiniGPT-4, which aligns a frozen visual encoder with a frozen advanced LLM, Vicuna, using one projection layer." (Abstract)
- "It consists of a vision encoder with a pretrained ViT and Q-Former, a single linear projection layer, and an advanced Vicuna large language model." (Figure 1 caption)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
