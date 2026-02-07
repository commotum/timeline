# Goal-Aware Neural SAT Solver (Not specified in the paper)
Source: Goal-Aware Neural SAT Solver (QuerySAT - goal-aware guidance).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAT solving (find satisfying variable assignment) | CNF Boolean formula as variables-clauses graph adjacency matrices A_p and A_n | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Variable assignment vector out in [0,1]^n | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper presents QuerySAT, a neural SAT solver that operates on CNF formulas encoded as variables-clauses graphs and outputs variable assignments to satisfy them. Across evaluated benchmarks (k-SAT, 3-SAT, 3-Clique, k-Coloring, SHA-1 preimage), the task remains SAT solving with the same input/output modality. The evidence supports 2D graph or adjacency-matrix inputs with capped sizes in experiments, dynamic query-driven processing, constructed recurrent state, and 1D assignment outputs.

## Evidence
### Task: SAT solving (find satisfying variable assignment)
- "finding a set of variable assignments that satisfies the given Boolean formula." (Section III-A)
- "It receives a CNF Boolean formula φ in the input represented as two adjacency matrices." (Section IV-A)
- "A_p ∈ {0,1}^{n × m} and A_n ∈ {0,1}^{n × m}" (Section IV-A)
- "The network outputs a vector out in [0,1]^n — a variable assignment." (Section IV-A)
- "at each step comes up with a query of variable assignments" (Introduction)
- "At the beginning an empty state vector initialized with all ones is allocated for each variable and each clause" (Section IV-A)
- "batch size of 20000 nodes (max node count in the input factor graph)" (Section IV-C)
- Inference: In Dimension set to 2D (x, y) based on the n × m adjacency matrices; Out Dimension set to 1D (t) based on the output vector out in [0,1]^n; In/Out Dynamics set to Capped based on the stated "max node count in the input factor graph"; Attention Dynamic set to Dynamic because the model issues a query "at each step"; State Dynamic set to Constructed because it allocates a "state vector" for each variable and clause. (Sections IV-A, IV-C, Introduction)
