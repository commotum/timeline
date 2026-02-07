# Learning a SAT Solver from Single-Bit Super-Vision (Not specified in the paper.)
Source: Learning a SAT Solver from Single-Bit Supervision (NeuroSAT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (satisfiable vs unsatisfiable) | SAT problem (CNF formula) | 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | satisfiability label | 0D (inferred) | Fixed (inferred) |
| generation (satisfying assignment) | SAT problem (CNF formula) | 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | satisfying assignment of truth values to variables | 1D (t) (inferred) | Open (inferred) |
| detection (unsat core) | SAT problem with an unsat core | 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | literals in the unsat core | 1D (t) (inferred) | Open (inferred) |
| classification (per-literal satisfiable-assignment existence) | SAT problem (CNF formula) | 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | per-literal bits indicating satisfiable-assignment existence | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper focuses on SAT problems (CNF formulas), framing satisfiability prediction as a classification task and also decoding satisfying assignments as solutions. It additionally trains a variant to detect unsat cores and reports a separate experiment predicting per-literal satisfiable-assignment existence. Based on the described graph/adjacency-matrix input and iterative message passing, the tasks operate over variable-size inputs with constructed state and static attention, producing either single-bit labels or variable-length literal/variable outputs.

## Evidence
### Task: classification (satisfiable vs unsatisfiable)
- "Classification task. For a SAT problem P, we define $\phi(P)$ to be true if and only if P is satisfiable." (Section 2 PROBLEM SETUP)
- "At test time, we get only the problem P and the goal is to predict $\phi(P)$, i.e. to determine if P is satisfiable." (Section 2 PROBLEM SETUP)
- "We provide NeuroSAT with only a single bit of supervision for each SAT problem that indicates whether or not the problem is satisfiable." (Section 1 Introduction)
- Inference: In Dimension 2D (x, y) and In Dynamics Open are inferred because the input is "any bipartite adjacency matrix M over any number of literals and clauses"; Attention/State are inferred from "iteratively refines a vector space embedding for each node by passing \"messages\" back and forth" (Section 3 Model). Out Dimension/Out Dynamics are inferred from the single-bit supervision.

### Task: generation (satisfying assignment)
- "the goal is to determine if the formula is satisfiable, and if so, to produce a satisfying assignment of truth values to variables." (Section 2 PROBLEM SETUP)
- "The solution itself can almost always be automatically decoded from the network's activations, making NeuroSAT an end-to-end SAT solver." (Section 1 Introduction)
- Inference: In Dimension 2D (x, y) and In Dynamics Open are inferred because the input is "any bipartite adjacency matrix M over any number of literals and clauses"; Attention/State are inferred from iterative message passing over the graph (Section 3 Model). Out Dimension 1D (t) and Out Dynamics Open are inferred from producing a per-variable satisfying assignment.

### Task: detection (unsat core)
- "it learns to detect these unsat cores instead of searching for satisfying assignments." (Section FINDING UNSAT CORES)
- "The literals involved in the unsat core can be decoded from its internal activations." (Section FINDING UNSAT CORES)
- Inference: In Dimension 2D (x, y) and In Dynamics Open are inferred because the input is "any bipartite adjacency matrix M over any number of literals and clauses"; Attention/State are inferred from iterative message passing over the graph (Section 3 Model). Out Dimension 1D (t) and Out Dynamics Open are inferred from outputting a subset of literals.

### Task: classification (per-literal satisfiable-assignment existence)
- "we also trained our architecture to predict whether there is a satisfying assignment involving each individual literal in the problem" (Section 10 DISCUSSION)
- "A SAT problem is a formula in CNF" (Section 2 PROBLEM SETUP)
- Inference: In Dimension 2D (x, y) and In Dynamics Open are inferred because the input is "any bipartite adjacency matrix M over any number of literals and clauses"; Attention/State are inferred from iterative message passing over the graph (Section 3 Model). Out Dimension 1D (t) and Out Dynamics Open are inferred from per-literal bit outputs.
