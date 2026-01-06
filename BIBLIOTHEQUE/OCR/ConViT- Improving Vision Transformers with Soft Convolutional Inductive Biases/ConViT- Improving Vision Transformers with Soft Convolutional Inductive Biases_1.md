# ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases (2021)
Source: ConViT- Improving Vision Transformers with Soft Convolutional Inductive Biases.md

## Core reasons
- Introduces gated positional self-attention (GPSA), explicitly controlling how positional information influences attention to add a soft convolutional bias in a transformer.
- Critiques standard positional self-attention and proposes modifications, centering the contribution on improving how position is handled in attention.

## Evidence extracts
- "To this end, we introduce gated positional self-attention (GPSA), a form of positional self-attention which can be equipped with a \"soft\" convolutional inductive bias. We initialize the GPSA layers to mimic the locality of convolutional layers, then give each attention head the freedom to escape locality by adjusting a gating parameter regulating the attention paid to position versus content information." (Abstract)
- "However, the standard parameterization of PSA layers (Eq. 4) suffers from two limitations, which lead us two introduce two modifications." (Section 3. Approach)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
