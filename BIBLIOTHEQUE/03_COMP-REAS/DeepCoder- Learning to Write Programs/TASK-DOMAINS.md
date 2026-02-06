# DEEPCODER: LEARNING TO WRITE PROGRAMS (Not specified in the paper)
Source: DeepCoder- Learning to Write Programs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis (inductive program synthesis) | input-output example pairs of integer arrays/integers | 0D;1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | program source code in DSL (sequence of function calls) | 1D (t) (inferred) | Capped (inferred) |
| Multi-label classification (program attribute/function prediction) | input-output example pairs of integer arrays/integers | 0D;1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | program attribute vector / function presence probabilities | 0D (inferred) | Fixed (inferred) |

## Summary
DeepCoder targets inductive program synthesis from input-output examples over integers and integer arrays, and its neural component performs multi-label prediction of which DSL functions appear in the target program. The paper’s inputs and outputs are 0D/1D structures (scalars and sequences), with capped sizes due to maximum array length and bounded program length, and the encoder’s fixed-length representation implies static attention. The synthesis pipeline constructs intermediate attribute predictions to guide search, while the attribute predictor itself is a direct mapping from examples to a fixed function-label vector.

## Evidence
### Task: Program synthesis (inductive program synthesis)
- "given input-output examples, produce a program that has behavior consistent with the examples." (Section 2 Background on Inductive Program Synthesis)
- "A program in our DSL is a sequence of function calls" (Section 4.1 Domain Specific Language and Attributes)
- "We use an optimized C++ implementation of depth-first search (DFS) to search over programs with a given maximum length T." (Appendix D Depth-First Search)
- "the result of each call initializes a fresh variable that is either a singleton integer or an integer array." (Section 4.1 Domain Specific Language and Attributes)
- "we pad the inputs and outputs to a maximum length L with a special NULL value." (Section 4.3 Machine Learning Model)
- "the machine learning model predicts a distribution  $q(\mathbf a \mid \mathcal E)$" (Section 3 Learning Inductive Program Synthesis)
- "the search procedure aims to search over programs P as ordered by  $q(\mathcal A(P) \mid \mathcal E)$ ." (Section 3 Learning Inductive Program Synthesis)
- "into a single (fixed-length) vector" (Section 4.3 Machine Learning Model)
- Inference: Inferred `0D;1D (t)` inputs and `Capped` input dynamics from inputs being a "singleton integer or an integer array" and padding to a "maximum length L"; inferred `1D (t)` and `Capped` outputs from a "sequence of function calls" and a "maximum length T"; inferred `Static` attention from the "fixed-length" encoding and `Constructed` state from attribute-guided search (model predicts `q(a|E)` and search is ordered by `q(A(P)|E)`).

### Task: Multi-label classification (program attribute/function prediction)
- "a machine learning model that maps from input-output examples to program attributes" (Section 3 Learning Inductive Program Synthesis)
- "The machine learning problem is to learn a distribution of attributes given input-output examples" (Section 3 Learning Inductive Program Synthesis)
- "predict the presence or absence of individual functions." (Section 4.3 Machine Learning Model)
- "represent the input and output types (singleton or array)" (Section 4.3 Machine Learning Model)
- "we pad the inputs and outputs to a maximum length L with a special NULL value." (Section 4.3 Machine Learning Model)
- "For the encoder we use a simple feed-forward architecture." (Section 4.3 Machine Learning Model)
- "C=34 is the number of functions in our DSL" (Section 4.3 Machine Learning Model)
- "into a single (fixed-length) vector" (Section 4.3 Machine Learning Model)
- Inference: Inferred `0D;1D (t)` inputs and `Capped` input dynamics from "singleton or array" typing and padding to a "maximum length L"; inferred `Static` attention from the "fixed-length" encoding; inferred `0D` outputs and `Fixed` output dynamics from a fixed set of function labels ("C=34"), and inferred `Direct` state from the feed-forward encoder/decoder mapping without any persistent state described.
