# Let's Verify Step by Step (Not specified in the paper)
Source: Let's Verify Step by Step.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Math problem solution generation | MATH problem statement (text tokens) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Step-by-step solution / final answer (text tokens) | 1D (t) | Not specified in the paper. |
| Solution correctness classification (outcome supervision) | Model-generated solution (text tokens) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Correct/incorrect label (solution-level score) | 0D | Fixed |
| Step-level correctness classification (process supervision) | Step-by-step solution or solution prefix (text tokens) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Positive/negative/neutral label per step (single token) | 0D | Fixed |

## Summary
The paper focuses on text-based mathematical problem solving on the MATH dataset, with models generating step-by-step solutions and final answers. It also trains reward models that classify correctness at the solution level (outcome supervision) and at the step level (process supervision). Inputs and generated solutions are token sequences (1D (t)), while verification outputs are scalar labels (0D) with fixed size; dynamics, attention, and state behaviors beyond that are not explicitly specified.

## Evidence
### Task: Math problem solution generation
- "Large language models are capable of solving tasks that require complex multistep reasoning by generating solutions in a step-by-step chain-of-thought format" (Section 1 Introduction)
- "we train the generator to produce solutions in a newline delimited step-by-step format. Specifically, we few-shot generate solutions to MATH training problems" (Section 2.3 Generator)

### Task: Solution correctness classification (outcome supervision)
- "we train the ORM to predict whether each solution is correct or incorrect" (Section 2.5 Outcome-supervised Reward Models (ORMs))
- "At test time, we use the ORM's prediction at the final token as the overall score for the solution" (Section 2.5 Outcome-supervised Reward Models (ORMs))

### Task: Step-level correctness classification (process supervision)
- "Their task is to assign each step in the solution a label of positive, negative, or neutral" (Section 2.4 Data Collection)
- "We train PRMs to predict the correctness of each step after the last token in each step. This prediction takes the form of a single token" (Section 2.6 Process-supervised Reward Models (PRMs))
