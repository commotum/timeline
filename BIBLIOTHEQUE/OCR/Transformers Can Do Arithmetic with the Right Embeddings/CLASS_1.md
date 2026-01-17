# Transformers Can Do Arithmetic with the Right Embeddings (2024)
Source: b75328-2024.pdf

## Core reasons
- The paper links transformer arithmetic failures to lost digit position information and proposes a direct fix to the representation.
- The main contribution is a new positional embedding (Abacus Embeddings) that encodes digit significance/relative position to improve generalization.

## Evidence extracts
- "Our experiments indicate that this difficulty stems from their inability to clearly represent the exact position of a digit within a long sequence of digits. To address this problem, we propose a simple modification to the data representation that directly addresses this shortcoming." (p. 2)
- "We call this Abacus Embeddings. We apply the same positional embedding to all digits of the same significance, providing an explicit signal that the model can use to align digits." (p. 5)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
