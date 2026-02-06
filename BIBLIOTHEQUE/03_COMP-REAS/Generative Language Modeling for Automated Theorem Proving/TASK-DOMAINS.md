# Generative Language Modeling for Automated Theorem Proving (Not specified in the paper)
Source: Generative Language Modeling for Automated Theorem Proving.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (proof steps/tactics) | goal statement (Metamath formal goal) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | proof step / tactic statement | 1D (t) (inferred) | Capped (inferred) |
| classification (goal provability / outcome) | goal statement (Metamath formal goal) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | OUTCOME label P/N (provable vs not) | 0D (inferred) | Fixed (inferred) |
| generation (full proofs via proof search) | root goal statement to prove (Metamath) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | proof (sequence of tactics/proof steps) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper targets automated theorem proving in Metamath, using a language model to generate proof steps and a learned value function to predict goal provability. Inputs and outputs are formal statement/proof-step sequences with capped lengths (context window and proof-search depth), yielding 1D sequence modalities, while provability outputs are 0D labels. Most model calls are static and direct, but the proof-search procedure constructs proof trees and dynamically selects which goals to expand.

## Evidence
### Task: generation (proof steps/tactics)
- "generate the PROOFSTEP given a GOAL" (Section 4.2 Training Objective)
- "GOAL <GOAL> PROOFSTEP <PROOFSTEP><EOT>" (Section 4.2 Training Objective)
- Inference: Treated inputs/outputs as 1D token sequences with capped dynamics and static/direct processing based on the language-modeling format and "a context size of 2048 tokens." (Sections 4.2 and 5)

### Task: classification (goal provability / outcome)
- "We implement the value function by means of an outcome objective" (Section 4.7 Learned Value Function)
- "GOAL <GOAL> OUTCOME <P|N><EOT>" (Section 4.7 Learned Value Function)
- "The binary nature of the OUTCOME allows the definition of a provability function" (Section 4.7 Learned Value Function)
- Inference: Interpreted the task as binary classification with 1D token input and a fixed 0D label output; capped dynamics follow from "a context size of 2048 tokens." (Sections 4.7 and 5)

### Task: generation (full proofs via proof search)
- "We find proofs by running proof searches." (Section 4.3.1 Goal Expansion)
- "A proof search maintains a proof tree and a queue of open goals" (Section 4.3.1 Goal Expansion)
- "Each successful tactic application generates new subgoals that are added to the proof tree and the proof search queue." (Section 4.3.1 Goal Expansion)
- "Each proof search involves d=128 goal expansions, so proofs generated have at most d proof steps." (Section 4.3.1 Goal Expansion)
- Inference: Classified this as proof-sequence generation with dynamic attention and constructed state because the procedure maintains a proof tree/queue and selects goals to expand; output length is capped by d. (Section 4.3.1)
