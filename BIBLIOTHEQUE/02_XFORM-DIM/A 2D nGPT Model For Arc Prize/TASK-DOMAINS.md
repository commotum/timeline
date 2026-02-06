# A 2D nGPT Model For Arc Prize (2024)
Source: A 2D nGPT Model For Arc Prize.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (grid-to-grid transformation) | colored grids (grid cell color indices) | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | colored grids | 2D (x, y) | Capped (inferred) |

## Summary
The paper focuses on ARC constant-size tasks that map input colored grids to output colored grids, with grids sized up to 30x30. This indicates a 2D grid domain with capped size variability and outputs matching input size. Based on the described architecture, attention is treated as static and state as direct (both inferred).

## Evidence
### Task: prediction (grid-to-grid transformation)
- "For each training sample we are given an input and a corresponding output." (Section 1 Introduction)
- "The problem is to compute the output of the test samples." (Section 1 Introduction)
- "Inputs and outputs are colored grids of dimension up to 30x30." (Section 1 Introduction)
- "takes as input a grid, and outputs a grid of same size." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- "the size of the output is always the same as the size of the input." (Section 1 Introduction)
- "A grid cell attends to all cells in same row, and it attends to all cell in same column." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Inference: In/Out Dynamics are Capped (inferred) because grids are "of dimension up to 30x30"; Attention Dynamic is Static (inferred) because attention is fixed to row/column; State Dynamic is Direct (inferred) because the model "takes as input a grid, and outputs a grid of same size." (Sections 1 Introduction, 2 A 2D nGPT Model for Constant Size Tasks)
