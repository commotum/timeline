# Solving olympiad geometry without human demonstrations (2024)
Source: Solving olympiad geometry without human demonstrations (AlphaGeometry).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Theorem proving (Euclidean plane geometry) | Geometry theorem premises and conclusion with evolving proof state | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Human-readable geometry proofs for theorem conclusions | 1D (t); 2D (x, y) (inferred) | Capped (inferred) |
| Auxiliary construction generation | Problem statement string and past constructions | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | One new auxiliary construction sentence | 1D (t); 2D (x, y) (inferred) | Capped (inferred) |
| Synthetic theorem and proof generation | Randomly sampled theorem premises | 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Synthetic (premises conclusion proof) examples and dependency subgraphs | 2D (x, y) (inferred) | Open (inferred) |

## Summary
The paper covers a neuro-symbolic geometry pipeline with three explicit tasks: generating synthetic theorem-proof data, generating auxiliary constructions, and solving Euclidean geometry theorem-proving problems end to end. The data objects are symbolic sequences grounded in plane-geometry entities, supporting 1D (t) and 2D (x, y) classifications where justified by text. Proof-search interfaces are capped by explicit context and iteration limits, whereas synthetic data generation is open-ended at the interface level through large-scale randomized sampling. The system behavior is state-constructive throughout, with dynamic runtime decision-making in proof search and static exhaustive deduction in the symbolic generator.

## Evidence
### Task: Theorem proving (Euclidean plane geometry)
- "We propose AlphaGeometry, a theorem prover for Euclidean plane geometry" (Abstract)
- "Notably, AlphaGeometry produces human-readable proofs" (Abstract)
- Inference: `1D (t); 2D (x, y)` is inferred from sequence processing ("We serialize (P, N, G(N)) into a text string") and explicit Euclidean-plane grounding ("We focus on Euclidean plane geometry") (Section "Training a language model on synthetic data"; Abstract). `Capped` is inferred from "We limit the maximum context length to 1,024 tokens" and "the maximum number of iterations is 16" (Methods sections "Language model architecture and training" and "Parallelized proof search"). `Dynamic` attention is inferred from "We use beam search to explore the top k constructions" (Section "Combining language modelling and symbolic engines"). `Constructed` state is inferred from "growing the proof state" (Fig. 1 caption).

### Task: Auxiliary construction generation
- "Any neural solver trained on our synthetic data, on the other hand, learns to perform auxiliary constructions from scratch without human demonstrations." (Section "Generating proofs beyond symbolic deduction")
- "The language model is seeded with the problem statement string and generates one extra sentence at each turn, conditioning on the problem statement and past constructions, describing one new auxiliary construction" (Section "Combining language modelling and symbolic engines")
- Inference: `1D (t); 2D (x, y)` is inferred from sentence-level generation over geometry constructions. `Capped` is inferred from the 1,024-token context limit and capped proof-search iterations (Methods sections "Language model architecture and training" and "Parallelized proof search"). `Dynamic` attention is inferred from runtime beam-search branching over alternatives (Section "Combining language modelling and symbolic engines"). `Constructed` state is inferred from iterative additions of auxiliary points that expand proof state (Fig. 1 caption).

### Task: Synthetic theorem and proof generation
- "We first sample a random set of theorem premises, serving as the input to the symbolic deduction engine" (Section "Synthetic theorems and proofs generation")
- "we extracted 100 million synthetic theorems and their proofs" (Abstract section after introduction)
- "we obtained a synthetic training example (premises, conclusion, proof) = (P, N, G(N))" (Section "Synthetic theorems and proofs generation")
- Inference: `2D (x, y)` is inferred because premises and deductions are over Euclidean point objects ("x are points objects") (Section "Synthetic theorems and proofs generation"). `Open` dynamics is inferred from unbounded randomized sampling at scale ("sampled nearly 1 billion" premises; "100,000 CPU workers" yielding "500 million" examples) (Section "Synthetic theorems and proofs generation"; Methods section "Parallelized data generation and deduplication"). `Static` attention is inferred from deterministic exhaustive closure behavior ("The engine exhaustively deduces new statements") (Fig. 1 caption). `Constructed` state is inferred from explicit construction of dependency DAGs and subgraphs (`G(N)`) (Section "Synthetic theorems and proofs generation").
