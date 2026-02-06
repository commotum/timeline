# End-to-End Differentiable Proving (Not specified in the paper.)
Source: End-to-End Differentiable Proving.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (KB completion / query proving) | knowledge base facts/rules and query atoms (predicate + terms) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | proof success score for a query/fact | 0D (inferred) | Fixed (inferred) |
| generation (logical rule induction) | knowledge base ground atoms (facts) and rule templates / prior rule-structure assumptions | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | induced function-free first-order logic rules | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper presents NTPs as differentiable provers for knowledge bases that score and predict facts/queries via multi-hop reasoning, and as a mechanism for inducing logical rules from KB facts. Inputs are symbolic atoms/rules and queries (1D sequences, inferred), producing scalar proof success scores (0D, inferred) or induced rule structures (1D, inferred). The reasoning uses bounded proof depth and predefined rule templates (capped, inferred) and relies on constructed proof states with max-based selection over proof paths (dynamic attention/state, inferred).

## Evidence
### Task: prediction (KB completion / query proving)
- "We introduce neural networks for end-to-end differentiable proving of queries to knowledge bases by operating on dense vector representations of symbols." (Abstract)
- "the resulting neural network can be trained to infer facts from a given incomplete knowledge base." (Abstract)
- "For all tasks, the goal is to predict locatedIn(e, e) for every test country e and all five regions e, but the access to training atoms in the KB varies." (5 Experiments)
- "While an atom is a list of a predicate symbol and terms" (3 Differentiable Prover)
- "A proof state  $S=(\psi,\rho)$  is a tuple consisting of the substitution set  $\psi$  constructed in the proof so far" (3 Differentiable Prover)
- "a neural network  $\rho$  that outputs a real-valued success score of a (partial) proof." (3 Differentiable Prover)
- "$\\underset{S \\neq \\mathsf{FAIL}}{\\arg\\max} S_{\\rho}$" (3.4 Proof Aggregation)
- "where d is a predefined maximum proof depth" (3.4 Proof Aggregation)
- Inference: Classified input/output dimensions and dynamics, attention, and state from atoms-as-lists, proof-state construction, real-valued success scores, argmax proof aggregation, and predefined maximum proof depth. (3 Differentiable Prover; 3.4 Proof Aggregation)

### Task: generation (logical rule induction)
- "induce logical rules" (Abstract)
- "We can use NTPs for ILP by gradient descent instead of a combinatorial search over the space of rules" (3.5 Neural Inductive Logic Programming)
- "to induce rules that let us prove known ground atoms" (3.5 Neural Inductive Logic Programming)
- "thereby inducing rules of predefined structures like the one above." (3.5 Neural Inductive Logic Programming)
- "a rule can be seen as a list of atoms and thus a list of lists" (3 Differentiable Prover)
- "A proof state  $S=(\psi,\rho)$  is a tuple consisting of the substitution set  $\psi$  constructed in the proof so far" (3 Differentiable Prover)
- "$\\underset{S \\neq \\mathsf{FAIL}}{\\arg\\max} S_{\\rho}$" (3.4 Proof Aggregation)
- Inference: Treated rules as 1D list-structured outputs with capped structure from predefined rule templates/structures, and carried over dynamic attention/constructed state from the NTP proof-state and argmax aggregation. (3 Differentiable Prover; 3.4 Proof Aggregation; 3.5 Neural Inductive Logic Programming)
