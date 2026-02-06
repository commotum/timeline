# A Fully First-Order Layer for Differentiable Optimization (2025)
Source: A Fully First-Order Layer for Differentiable Optimization.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (decision-focused learning) | input feature vectors x_i | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | QP decision vector y*(theta, x_i) | 1D (t) (inferred) | Fixed (inferred) |
| prediction (Sudoku solution) | partially filled Sudoku puzzle grid p_i | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Sudoku solution grid y*(theta, p_i) | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper evaluates the first-order differentiable optimization layer on two tasks: a synthetic decision-focused learning setup mapping fixed-length feature vectors to QP decision vectors, and Sudoku puzzle completion mapping partially filled n x n grids to LP solutions. The inputs and outputs span 1D and 2D domains, with fixed dynamics implied by the fixed vector sizes and the stated n=9 grid size. Attention and state dynamics are not specified for these tasks.

## Evidence
### Task: prediction (decision-focused learning)
- "Given the dataset of ground-truth outputs  $y_i \in \mathbb{R}^{d_y}$  and inputs  $x_i \in \mathbb{R}^{d_x}$" (Section 6.1 Synthetic Decision-Focused Task)
- "the solution to the quadratic program  $y^*(\theta, x_i)$  is fed into a linear loss function  $y_i^{\top}y^*(\theta, x_i)$ ." (Section 6.1 Synthetic Decision-Focused Task)
- "The task is to learn the neural network parameter  $\theta$  to minimize  $\sum_{i=1}^{N} y_i^{\top}y^*(\theta, x_i)$ ." (Section 6.1 Synthetic Decision-Focused Task)
- Inference: In/Out Dimension and Dynamics are labeled 1D and Fixed because inputs and outputs are vectors in $\mathbb{R}^{d_x}$ and $\mathbb{R}^{d_y}$, indicating fixed-length arrays.

### Task: prediction (Sudoku solution)
- "Given a dataset of partially filled  $n \times n$  Sudoku puzzles  $p_i \in \{0,1\}^{n^3}$  and their solutions  $y_i \in \{0,1\}^{n^3}$" (Section 6.2 Sudoku Task)
- "the task is to learn the rules of Sudoku puzzles, which are the linear constraint parameters  $A(\theta)$  and  $b(\theta)$  of the linear program." (Section 6.2 Sudoku Task)
- "Here, we set n=9" (Section 6.2 Sudoku Task)
- Inference: In/Out Dimension and Dynamics are labeled 2D and Fixed because the task is defined on n x n Sudoku grids with n=9.

## CSV Output (required)
Write a CSV file to "/home/jake/Developer/timeline/BIBLIOTHEQUE/05_ML-FNDTNS/A Fully First-Order Layer for Differentiable Optimization/.TASK-DOMAINS.csv.tmp.84e3467ae8694ccc898928cd6965e78a" with the same rows and columns as the Task
Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
