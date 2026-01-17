# EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION (Not specified in the paper)
Source: Extending Context Window of Large Language Models via Positional Interpolation (PI).md

## Core reasons
- The paper critiques direct length extrapolation with RoPE and ties failures to positional encoding behavior.
- The main contribution is a change to positional encoding by interpolating/rescaling position indices to stay within the pretrained range.

## Evidence extracts
- "While certain techniques such as ALiBi (Press et al., 2022) and LeX (Sun et al., 2022) enable length extrapolation of Transformers, i.e. train on short context windows and inference on longer ones, many existing pre-trained LLMs, including LLaMA (Touvron et al., 2023), use positional encodings that have weak extrapolation properties (e.g., RoPE (Su et al., 2021))." (Section 1 Introduction)
- "Formally, we replace RoPE f by f' defined as follows

$$\mathbf{f}'(\mathbf{x}, m) = \mathbf{f}\left(\mathbf{x}, \frac{mL}{L'}\right). \tag{4}$$

We call this transformation on the position encoding **Position Interpolation**." (Section 2.3 Proposed Approach: Position Interpolation)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
