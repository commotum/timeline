# Formal Mathematics Statement Curriculum Learning (Not specified in the paper.)
Source: Formal Mathematics Statement Curriculum Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tactic generation (proofstep prediction) | Lean tactic state (goal) and declaration name | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Lean tactic / proofstep | 1D (t) (inferred) | Not specified in the paper. |
| Proofsize bucket prediction | Lean tactic state (goal) and declaration name | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Proofsize bucket token | 0D (inferred) | Fixed (inferred) |
| Formal theorem proving (proof search) | Formal statement to prove (Lean statement) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Proof / proof search trajectory (sequence of tactics) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper defines two language-model objectives in Lean: generating tactics from tactic states and predicting proofsize buckets, and it uses these within proof search to solve formal statements. Inputs and outputs are formal text/tactic sequences (1D (t)), while proofsize prediction outputs a single label (0D); the paper does not specify interface size limits. Proof search uses best-first search with a growing proof tree, implying dynamic attention and constructed state (inferred).

## Evidence
### Task: Tactic generation (proofstep prediction)
- "consists in generating a PROOFSTEP (a Lean tactic) given a GOAL (a Lean tactic state)." (Section 4.3.1)
- "We also condition this objective on the current DECLARATION (a Lean theorem name)" (Section 4.3.1)
- Inference: Assigned 1D (t) to input/output because the objective is serialized as "DECL <DECLARATION>GOAL <GOAL> PROOFSTEP <PROOFSTEP>." (Section 4.3.1)

### Task: Proofsize bucket prediction
- "consists in generating one token that represents a proof size estimate bucket" (Section 4.3.2)
- "DECL <DECLARATION> GOAL <GOAL> PROOFSIZE <PROOFSIZE_BUCKET_TOKEN>" (Section 4.3.2)
- Inference: Output set to 0D/Fixed because it is "generating one token"; input set to 1D (t) due to the linear sequence format. (Section 4.3.2)

### Task: Formal theorem proving (proof search)
- "Our expert iteration process takes as input: (i) a set of formal statements St" (Section 4.5)
- "sampling proof searches for statements in St using θ_k" (Section 4.5)
- "whether a trajectory (i.e. a proof) is successful" (Introduction)
- Inference: Set input/output to 1D (t) because proofs are described as a "trajectory" (Introduction); set Attention to Dynamic due to "best-first search" and State to Constructed because they grow a "unique proof tree" (Section 4.5).
