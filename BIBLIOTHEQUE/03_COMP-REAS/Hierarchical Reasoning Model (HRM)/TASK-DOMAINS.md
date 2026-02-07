# Hierarchical Reasoning Model (Not specified in the paper.)
Source: Hierarchical Reasoning Model (HRM).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (ARC-AGI grid transformation) | input-output demonstration grid pairs + test input grid | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | output grid | 2D (x, y) | Capped (inferred) |
| prediction (Sudoku solution) | partially filled 9x9 Sudoku grid (digits 1-9) | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | completed 9x9 Sudoku solution grid | 2D (x, y) | Fixed |
| prediction (optimal path) | 30x30 maze grid (start-to-goal) | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | optimal path through the maze (path cells) | 2D (x, y) | Fixed |

## Summary
The paper evaluates HRM on three grid-based reasoning tasks: ARC-AGI inductive puzzle solving, Sudoku-Extreme grid completion, and Maze-Hard optimal path finding. Inputs and outputs are 2D grids, with fixed-size grids for Sudoku (9x9) and Maze (30x30), and capped-size grids for ARC-AGI due to padding to a maximum sequence length. The model uses recurrent hidden states and fixed token sequences, so Attention is treated as Static and State as Constructed where supported by the architectural description (inferred).

## Evidence
### Task: prediction (ARC-AGI grid transformation)
- "The initial version, ARC-AGI-1, presents challenges as input-label grid pairs" (Section 3.1 Benchmarks)
- "Each task provides a few input-output demonstration pairs" (Section 3.1 Benchmarks)
- "An AI model has two attempts to produce the correct output grid." (Section 3.1 Benchmarks)
- Inference: Set In/Out Dynamics to Capped and Attention to Static because "The two-dimensional input and output grids were flattened and then padded to the maximum sequence length." (Section 3.2 Evaluation Details). Set State to Constructed because "The modules  $f_L$  and  $f_H$  each keep a hidden state" (Section 2 Hierarchical Reasoning Model).

### Task: prediction (Sudoku solution)
- "Sudoku is a  $9\times9$  logic puzzle" (Section 3.1 Benchmarks)
- "A prediction is considered correct if it exactly matches the puzzle's unique solution." (Section 3.1 Benchmarks)
- Inference: Attention set to Static because "The two-dimensional input and output grids were flattened and then padded to the maximum sequence length." (Section 3.2 Evaluation Details). State set to Constructed because "The modules  $f_L$  and  $f_H$  each keep a hidden state" (Section 2 Hierarchical Reasoning Model).

### Task: prediction (optimal path)
- "This task involves finding the optimal path in a  $30 \times 30$  maze" (Section 3.1 Benchmarks)
- "A path is considered correct if it is valid and optimal" (Section 3.1 Benchmarks)
- Inference: Attention set to Static because "The two-dimensional input and output grids were flattened and then padded to the maximum sequence length." (Section 3.2 Evaluation Details). State set to Constructed because "The modules  $f_L$  and  $f_H$  each keep a hidden state" (Section 2 Hierarchical Reasoning Model).
