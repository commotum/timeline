# Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge (Not specified in the paper.)
Source: Logic Tensor Networks- Deep Learning and Logical Reasoning from Data and Knowledge.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knowledge completion | knowledge-base facts and logical constraints over objects/relations | 0D (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | truth-values for known and missing facts; groundings (feature vectors) for constants | 0D; 1D (t) (inferred) | Open (inferred) |
| data prediction (numerical properties or class) | object groundings (real-valued vectors) and relational structure | 0D; 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | numerical properties (real-valued vectors) or class labels for objects | 0D; 1D (t) (inferred) | Open (inferred) |

## Summary
The paper covers knowledge completion over relational knowledge bases and data prediction of numerical properties or class labels for objects. Inputs span symbolic relational facts and real-valued vectors, and outputs include truth-values for facts plus predicted feature vectors or class labels. The described open-domain setting implies 0D and 1D structures with Open dynamics and Constructed state (inferred), while attention dynamics are not specified.

## Evidence
### Task: knowledge completion
- "illustrate the task of knowledge completion in 1tn." (Section 5 An Example of Knowledge Completion)
- "Our main task is to complete the knowledge-base (KB)" (Section 5 An Example of Knowledge Completion)
- "find a truth-value for all the missing facts, e.g. C(i)" (Section 5 An Example of Knowledge Completion)
- "find the grounding of each constant symbol a, ..., n." (Section 5 An Example of Knowledge Completion)
- Inference: Marked In/Out Dynamics as Open because the agent manages "an unbounded, possibly infinite, set of objects  $O = {o_1, o_2, \ldots}$ ." (Section 1 Introduction) and existentially quantified formulas can be satisfied by "new individuals" (Section 6). Marked State and 1D output because groundings are "represented by an n-tuple of real values  $\mathcal{G}(o_i) \in \mathbb{R}^n$" (Section 1 Introduction) and the task must "find the grounding of each constant symbol a, ..., n." (Section 5 An Example of Knowledge Completion)

### Task: data prediction (numerical properties or class)
- "predict the numerical properties or the class of the objects in O." (Section 1 Introduction)
- "represented by an n-tuple of real values  $\mathcal{G}(o_i) \in \mathbb{R}^n$" (Section 1 Introduction)
- "LTN allows one to generate data for prediction." (Section 6)
- Inference: Marked In/Out Dynamics as Open because the setting involves "an unbounded, possibly infinite, set of objects  $O = {o_1, o_2, \ldots}$ ." (Section 1 Introduction) and existentially quantified formulas can be satisfied by "new individuals" (Section 6). Marked State and 1D dimensions as Constructed because prediction uses learned groundings that are "represented by an n-tuple of real values  $\mathcal{G}(o_i) \in \mathbb{R}^n$." (Section 1 Introduction)
