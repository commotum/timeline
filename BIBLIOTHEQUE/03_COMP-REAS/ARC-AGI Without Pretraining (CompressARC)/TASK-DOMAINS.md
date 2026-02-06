# ARC-AGI WITHOUT PRETRAINING (Not specified in the paper)
Source: ARC-AGI Without Pretraining (CompressARC).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (output grid) | colored grid pairs (example input/output grids; test input grid) | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | target colored output grid(s) | 2D (x, y) | Capped (inferred) |

## Summary
CompressARC is framed around solving ARC-AGI visual reasoning puzzles by predicting target colored output grids from example input/output grid pairs. The task operates over 2D grid data, with variable grid shapes handled via per-puzzle size selection (dynamics capped, inferred). The paper describes a fixed, full-grid processing architecture and multitensor internal representations, suggesting static attention and constructed state (both inferred).

## Evidence
### Task: prediction (output grid)
- "apply to an input colored grid to produce a ground truth target colored grid." (Section 2 BACKGROUND: THE ARC-AGI BENCHMARK)
- "Several input-output grid pairs are given as examples to help the system figure out the hidden rule in the puzzle." (Section 2 BACKGROUND: THE ARC-AGI BENCHMARK)
- "Each puzzle takes the form of a tensor of shape [n_example, width, height, 2]." (Section 3, Puzzle/solution data format)
- "you get two guesses to guess the output grid(s) for a given input grid." (Section L, Scoring)
- "The raw data consists of grids of various shapes, while the neural network operates on grids of constant shape." (Section F.1 OUTPUT SHAPE DETERMINATION)
- "we make a temporary prediction of the largest width and height out of the grids in the given ARC-AGI puzzle." (Section F.1 OUTPUT SHAPE DETERMINATION)
- "outputs a [n_example, n_colors, width, height, 2]-shaped logit tensor" (Section 3.1 RESTRICTING THE PROGRAM SPACE)
- "Both the input z to the network and the outputted logits, as well as all of the internal activations, take the form of a multitensor." (Section C MULTITENSORS)
- "consists of a decoding layer functioning like an embedding matrix (details in Appendix D.1), followed by a core with a residual backbone" (Section 4 ARCHITECTURE)
- Inference: Marked In/Out Dynamics as Capped (inferred) because grids are described as various shapes and the largest width/height in a puzzle is used as the working size (Section F.1 OUTPUT SHAPE DETERMINATION). Marked Attention Dynamic as Static (inferred) because the model outputs full-grid logits with fixed tensor shapes rather than selecting inputs at runtime (Section 3.1). Marked State Dynamic as Constructed (inferred) because the method maintains internal multitensor activations and a residual backbone (Sections C and 4).
