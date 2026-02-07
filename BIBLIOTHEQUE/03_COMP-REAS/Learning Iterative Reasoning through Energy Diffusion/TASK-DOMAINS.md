# Learning Iterative Reasoning through Energy Diffusion (2024)
Source: Learning Iterative Reasoning through Energy Diffusion.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matrix addition | two 20x20 matrices | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | summed 20x20 matrix | 2D (x, y) (inferred) | Fixed (inferred) |
| matrix completion | partially observed low-rank 20x20 matrix (50% entries masked) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | completed 20x20 matrix | 2D (x, y) (inferred) | Fixed (inferred) |
| matrix inversion | 20x20 matrix | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | inverse 20x20 matrix | 2D (x, y) (inferred) | Fixed (inferred) |
| Sudoku solving | partially filled Sudoku board (zeros for unknowns) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | valid Sudoku solution board | 2D (x, y) (inferred) | Fixed (inferred) |
| visual Sudoku solving | image of Sudoku grid with MNIST digits | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Sudoku solution board (inferred) | 2D (x, y) (inferred) | Fixed (inferred) |
| graph connectivity prediction | graph adjacency matrix | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | graph connectivity matrix | 2D (x, y) (inferred) | Capped (inferred) |
| shortest-path planning | directed graph adjacency matrix + start/goal node embeddings | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | plan/action sequence matrix [T, N] | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper evaluates IRED on continuous matrix computation tasks (addition, completion, inversion), discrete grid-based reasoning (Sudoku and visual Sudoku), and graph-based reasoning/planning (connectivity and shortest-path planning). Inputs and outputs are predominantly 2D structures (matrices, boards, images), with fixed-size dynamics for matrix and Sudoku tasks and capped-size dynamics for variable-size graphs and plans. Attention is treated as static and state as constructed (inferred) because inference repeatedly optimizes candidate solutions over multiple steps given fixed inputs.

## Evidence
### Task: matrix addition
- "We consider three matrix operations on  $20 \times 20$  matrices," (Section 4.1)
- "Addition: We first evaluate neural networks in their ability to add matrices together (element-wise)." (Section 4.1)
- "Inputs and outputs are 20 by 20 matrices." (Table 1 caption)
- Inference: Mapped matrices to 2D (x, y) and Fixed dynamics based on the 20x20 matrix statements; attention/state inferred from iterative optimization with fixed input ("Input: Input task x_i"; "The final output of y^T is obtained after T steps of optimization.") (Section 4.1; Algorithm 2; Section 3.1)

### Task: matrix completion
- "We consider three matrix operations on  $20 \times 20$  matrices," (Section 4.1)
- "Matrix Completion: Next, we evaluate neural networks on their ability to do low-rank matrix completion." (Section 4.1)
- "We mask out 50% of the entries of a low-rank input matrix" (Section 4.1)
- "Inputs and outputs are 20 by 20 matrices." (Table 1 caption)
- Inference: Mapped matrices to 2D (x, y) and Fixed dynamics from the 20x20 matrix statements; attention/state inferred from iterative optimization with fixed input ("Input: Input task x_i"; "The final output of y^T is obtained after T steps of optimization.") (Section 4.1; Algorithm 2; Section 3.1)

### Task: matrix inversion
- "We consider three matrix operations on  $20 \times 20$  matrices," (Section 4.1)
- "Matrix Inverse: Finally, we evaluate neural networks on their ability to compute matrix inverses." (Section 4.1)
- "Inputs and outputs are 20 by 20 matrices." (Table 1 caption)
- Inference: Mapped matrices to 2D (x, y) and Fixed dynamics from the 20x20 matrix statements; attention/state inferred from iterative optimization with fixed input ("Input: Input task x_i"; "The final output of y^T is obtained after T steps of optimization.") (Section 4.1; Algorithm 2; Section 3.1)

### Task: Sudoku solving
- "the model is given a partially filled Sudoku board, with 0's filled-in entries that are currently unknown." (Section 4.2)
- "The task is to predict a valid solution that jointly satisfies the Sodoku rules and that is consistent with the given numbers." (Section 4.2)
- Inference: Treated the board/solution as 2D (x, y) with Fixed dynamics (standard fixed-size Sudoku grid implied by the board description); attention/state inferred from iterative optimization with fixed input ("Input: Input task x_i"; "The final output of y^T is obtained after T steps of optimization.") (Section 4.2; Algorithm 2; Section 3.1)

### Task: visual Sudoku solving
- "consists of MNIST digits written on a grid." (Section 4.2, Extension to Visual Sudoku)
- "The task is to predict a valid solution that jointly satisfies the Sodoku rules and that is consistent with the given numbers." (Section 4.2)
- Inference: Treated the image/grid input as 2D (x, y) with Fixed dynamics; output assumed to be the Sudoku solution board described in the Sudoku task; attention/state inferred from iterative optimization with fixed input ("Input: Input task x_i"; "The final output of y^T is obtained after T steps of optimization.") (Section 4.2; Algorithm 2; Section 3.1)

### Task: graph connectivity prediction
- "the model is given the adjacency matrix of a graph (1 if there is an edge directly connecting two nodes)." (Section 4.2)
- "The task is to predict the connectivity matrix of the graph (1 if there is a path connecting two nodes)." (Section 4.2)
- "training and standard test sets contain graphs with at most 12 nodes and our harder dataset contains graphs with 18 nodes." (Section 4.2)
- Inference: Mapped adjacency/connectivity matrices to 2D (x, y) and Capped dynamics based on max node counts; attention/state inferred from iterative optimization with fixed input ("Input: Input task x_i"; "The final output of y^T is obtained after T steps of optimization.") (Section 4.2; Algorithm 2; Section 3.1)

### Task: shortest-path planning
- "finding the shortest path in a graph." (Section 4.3)
- "the input to the model is the adjacency matrix of a directed graph," (Section 4.3)
- "The task is to predict a sequence of actions corresponding to the plan." (Section 4.3)
- "the output is a matrix of size [T, N]," (Section 4.3)
- "harder tasks consists of graphs of size 25 while models are trained on graphs of size 15." (Table 6 caption)
- Inference: Treated adjacency/plan matrices as 2D (x, y) with Capped dynamics from graph-size limits; attention/state inferred from iterative optimization with fixed input ("Input: Input task x_i"; "The final output of y^T is obtained after T steps of optimization.") (Section 4.3; Algorithm 2; Section 3.1)
