# SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver (2019)
Source: SATNet- Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Parity prediction (chained XOR) | Binary bit sequences | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Parity bit | 0D (inferred) | Fixed (inferred) |
| Sudoku solution completion (logical board) | Partially filled Sudoku board in logical bit form, plus mask of unknown bits | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Completed Sudoku board in logical bit form | 2D (x, y) (inferred) | Fixed (inferred) |
| Visual Sudoku solving (image to logical board solution) | Image representation of a Sudoku board (MNIST digits) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Logical Sudoku board solution | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper evaluates SATNet on three task domains: parity prediction, logical Sudoku completion, and visual Sudoku image-to-logical-solution mapping. The supported input/output structures span 1D ordered bit sequences and 2D board/image domains, with outputs that are either a single parity decision or full Sudoku board completions. Across the reported setups, task interfaces are fixed-size (sequence length $L$ chosen per model and fixed-size Sudoku boards), attention is static, and state is constructed through SATNet's continuous relaxations, auxiliary variables, and iterative coordinate-descent inference.

## Evidence
### Task: Parity prediction (chained XOR)
- "This experiment tests SATNet's ability to differentiate through many successive SAT problems by learning to compute the parity function." (Section 4.1)
- "The task is to map input sequences to their parity, given a dataset of example sequence/parity pairs." (Section 4.1)
- "Hence, for a sequence of length L, we construct our model to contain a sequence of L-1 SATNet layers with tied weights (similar to a recurrent network)." (Section 4.1)
- Inference: `1D (t) (inferred)` and `0D (inferred)` follow from "input sequences" and "single-bit supervision"; `Fixed (inferred)` follows from "for a sequence of length L" with model instances built for specific L; `Static (inferred)` is supported by the fixed layerwise input flow; `Constructed (inferred)` is supported by chained intermediate outputs ("layer d receives value d along with the rounded output of layer d-1") and SATNet's auxiliary inference state ("auxiliary variables ... register memory that is useful for inference"). (Section 4.1; Section 3.2.1)

### Task: Sudoku solution completion (logical board)
- "In Sudoku, given a (typically)  $9 \times 9$  partially-filled grid of numbers, a player must fill in the remaining empty grid cells..." (Section 4.2)
- "We construct a SATNet model for this task that takes as input a logical (bit) representation of the initial Sudoku board" (Section 4.2)
- "Given this input, the SATNet layer then outputs a bit representation of the Sudoku board with guesses for the unknown bits." (Section 4.2)
- Inference: `2D (x, y) (inferred)` is supported by the explicit " $9 \times 9$  ... grid" board structure; `Fixed (inferred)` is supported by fixed board size and a single SATNet layer for the task; `Static (inferred)` is supported by direct board+mask processing without runtime input selection; `Constructed (inferred)` is supported by SATNet's iterative inference and explicit auxiliary variables used as memory-like structure. (Section 4.2; Section 3.2.1; Section 3.2.3)

### Task: Visual Sudoku solving (image to logical board solution)
- "Specifically, we solve the visual Sudoku problem: that is, given an *image representation* of a Sudoku board ... constructed with MNIST digits, our network must output a *logical solution* to the associated Sudoku problem." (Section 4.3)
- "Each cell-wise probabilistic output of this convolutional layer is then fed as logical input to the SATNet layer, along with an input mask..." (Section 4.3)
- "The whole model is trained end-to-end..." (Section 4.3)
- Inference: `2D (x, y) (inferred)` is supported by the image-board input and board-structured logical solution; `Fixed (inferred)` is supported by fixed board/task setup in experiments; `Static (inferred)` is supported by the fixed convolution-to-SATNet pipeline; `Constructed (inferred)` is supported by constructed probabilistic features from LeNet and SATNet's internal iterative logical inference state. (Section 4.3; Section 3.2.3)
