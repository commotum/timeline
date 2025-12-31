# A2DnGPTModelForArcPrize (2024)
Source: 9908df-2024.pdf

## Core reasons
- The paper builds A2DnGPT by adapting a GPT-like transformer to ARC grids: the model reads and writes 2D inputs, replacing causal attention with row/column attention so each cell attends along its row and column before decoding colors (p. 3).
- Symmetry and color-permutation augmentations plus invertible logits transformations expand the transformer’s coverage of ARC’s 2D tasks, showing the core contribution is adapting the architecture to the spatial domain rather than repurposing positional encodings (p. 5).

## Evidence extracts
- "We use a transformer that takes as input a grid, and outputs a grid of same size. This transformer is similar to LLMs, GPT2 in particular, except it works with 2D grids rather than 1D sequences. Tokens are replaced by grid cell color indices. There are 10 colors in ARC tasks, plus an 11th color for padding grids to the same dimension when batching.
 The architecture is the same as a LLM:
 • an embedding layer that turns color indices into vectors,
 • a number of attention + feed forward layers,
 • a decoding layer producing color logits.
 The difference with LLMs is in the attention layers. The attention layers are 2D attention. A grid cell attends to all cells in same row, and it attends to all cell in same column. Each of these attention is a 1D attention and is implemented using pytorch masked scaled dot product attention." (p. 3)
- "A second way to increase tasks is to use invertible transformations. For each task, we could create another task by applying either a 2D symmetry (rotation, transpose) or a color symmetry (permutation). There are 8 2D symmetry, and, depending on the task up to 10 factorial relevant color permutations." (p. 5)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
