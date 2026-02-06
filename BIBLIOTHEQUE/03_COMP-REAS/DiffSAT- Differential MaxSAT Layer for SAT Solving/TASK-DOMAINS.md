# DiffSAT: Differential MaxSAT Layer for SAT Solving (Not specified in the paper)
Source: DiffSAT- Differential MaxSAT Layer for SAT Solving.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Satisfying assignment search (SAT solving) | CNF formula (clauses/literals) | 2D (x, y) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Boolean variable assignment that satisfies CNF | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper focuses on SAT solving by finding satisfying assignments for CNF formulas rather than unsatisfiable certification. Inputs are CNF clause/literal structures and outputs are Boolean variable assignments, implying 2D input structure and 1D output vectors (inferred from the clause matrix and assignment notation). The method supports varying problem sizes and uses iterative, gradient-driven updates with dynamic variable selection and constructed state during search (inferred).

## Evidence
### Task: Satisfying assignment search (SAT solving)
- "DiffSAT aims to determine the satisfying assignment for a given CNF formula." (Section 4.1, Evaluation Metric)
- "The accuracy of DiffSAT is evaluated based on whether the variable assignments it produces can satisfy the given constraints." (Section 4.1, Evaluation Metric)
- "In the forward pass, the inputs consist of relaxed solutions and the given CNFs  $\phi$ ." (Section 3.3, Differential MaxSAT Layer)
- Inference: In Dimension set to 2D (x, y) and In/Out Dynamics set to Open because clauses are represented as a matrix with variable counts, e.g., "The m clauses can also be represented as a clause matrix  $S \in \{1, -1, 0\}^{m \times n}$" and CNFs are described with "*n* binary variables and *m* clauses" (Section 3.1). Out Dimension set to 1D (t) because assignments are vectors, "Variable assignment  $v^k \in \mathbb{R}^n$  at k-th epoch" (Algorithm 1). Attention Dynamic marked Dynamic because the backward pass selects variables from falsified clauses, "we obtain the set of variables  $\bar{I}$  that are present in the falsified clauses" and "we proceed to select the best variable from this set using a criterion based on its gradient" (Section 3.3). State Dynamic marked Constructed because the solver "iteratively refin[es] variable assignments using gradient descent" (Introduction).
