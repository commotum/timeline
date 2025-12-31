# Extending Context Window of Large Language Models from a Distributional Perspective (2024)
Source: ad8ea4-2024.pdf

## Core reasons
- The paper extends LLM context windows by applying position interpolation strategies on rotary position embedding (RoPE), making positional encoding the main lever.
- It introduces a distribution-based method for selecting extension strategies that minimize perturbation of rotary angle distributions, i.e., a direct positional encoding improvement for long-context use.

## Evidence extracts
- "the context window is dynamically extended via fine-tuning or tuning-free position interpolation strategies (Chen et al., 2023; Peng et al., 2023; Liu et al., 2023) on the rotary position embedding." (p. 2)
- "Hence, we propose a method for length extension strategy selection, which has the potential to be the- oretically optimal by minimizing the perturbation to the rotary angle distributions of the pre-trained language model." (p. 2)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
