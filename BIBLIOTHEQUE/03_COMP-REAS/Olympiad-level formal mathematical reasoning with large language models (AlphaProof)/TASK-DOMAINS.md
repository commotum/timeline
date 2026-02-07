# Olympiad-level formal mathematical reasoning with reinforcement learning (2025)
Source: Olympiad-level formal mathematical reasoning with large language models (AlphaProof).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (tactic selection for formal theorem proving/disproof) | Lean tactic state (hypotheses and goals) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Lean tactic (text string) | 1D (t) | Not specified in the paper. |
| Prediction (value / expected return estimation) | Lean tactic state | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Value estimate of expected return / proof difficulty | 0D | Not specified in the paper. |
| Generation (auto-formalization: natural language -> Lean) | Natural language / LaTeX math problem statement | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Lean formal statement (code) | 1D (t) | Not specified in the paper. |
| Generation (synthetic Lean problem variants for TTRL) | Target Lean problem statement (formal instance T) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Synthetic Lean variants (formal problem statements) | 1D (t) | Not specified in the paper. |
| Generation (candidate answer guessing) | Formalized problems requiring an answer (with placeholders) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Candidate answers | 1D (t) | Not specified in the paper. |

## Summary
The paper covers formal theorem proving in Lean via tactic selection (and associated value estimation), along with auxiliary generation tasks for auto-formalization, variant creation for TTRL, and candidate answer guessing. Inputs and outputs are predominantly text-like sequences (1D (t)) in both natural language and Lean code, with one scalar value output (0D). Most dynamics and attention/state properties are not explicitly specified; the only inferred property is a constructed state for proof search due to explicit tree search over proof states.

## Evidence
### Task: Control (tactic selection for formal theorem proving/disproof)
- "the state s_t is the logical state of the Lean prover, encompassing established hypotheses and remaining goals, observed by the agent as the Lean tactic state" (Section 1.1 The Lean RL Environment)
- "The agent interacts by proposing an action a_t, a Lean tactic, as a text string." (Section 1.1 The Lean RL Environment)
- "randomly assigns the objective as either proving or disproving the statement." (Section 5.3.3 Matchmaker System)
- Inference: State Dynamic marked Constructed because the system builds a search tree over proof states: "the search iteratively expands the tree." (Figure 1 caption)

### Task: Prediction (value / expected return estimation)
- "produces two outputs: a list of N promising tactics to try and a value estimating proof difficulty." (Figure 1 caption)
- "The value head is trained to predict the return obtained at the current proof (sub)goal." (Section 5.3.3 Learner and Network Updates)

### Task: Generation (auto-formalization: natural language -> Lean)
- "This process translates mathematical statements from natural language into the formal language of Lean" (Section 5.3.2 Auto-Formalization)
- "receives the LaTeX-formatted natural language text of a problem statement, and outputs formal statements as valid Lean code." (Extended Data Figure 2)

### Task: Generation (synthetic Lean problem variants for TTRL)
- "generation of a problem-specific curriculum of synthetic variants for each target Lean instance T." (Section 5.4.2 Scaling with TTRL)
- "yielded hundreds of thousands of unique, syntactically-valid Lean variants (V_T) for each target T." (Section 5.4.2 Scaling with TTRL)

### Task: Generation (candidate answer guessing)
- "the system was responsible for generating candidate answers" (Section 6.4 IMO 2024 Evaluation Protocol and Methods)
- "We generated k=500 answers, successfully guessing the correct answers for all applicable problems." (Section 6.4 IMO 2024 Evaluation Protocol and Methods)
