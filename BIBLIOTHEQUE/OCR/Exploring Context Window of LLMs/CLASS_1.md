# Exploring Context Window of Large Language Models via Decomposed Positional Vectors (2024)
Source: a45446-2024.pdf

## Core reasons
- The paper frames context-window failures as an out-of-distribution positional encoding problem in Transformer-based LLMs.
- It proposes positional-vector-based context window extension methods (positional vector replacement and attention window extension) to improve how position is handled beyond the original window.

## Evidence extracts
- "Beyond the context window, the positional encodings at larger position indices are out-of-distribution (OOD), not encountered during the training phase." (p. 1)
- "Inspired by our analysis of the formation of positional vectors and the interpolation of positional vectors when extending the context window, we propose two training-free context window extension methods, i.e., positional vector replacement and attention window extension." (p. 8)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
