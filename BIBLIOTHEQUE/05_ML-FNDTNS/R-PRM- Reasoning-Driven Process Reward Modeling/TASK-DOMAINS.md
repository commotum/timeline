# R-PRM: Reasoning-Driven Process Reward Modeling (Not specified in the paper.)
Source: R-PRM- Reasoning-Driven Process Reward Modeling.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Process-level step evaluation (analysis + correctness judgment / reward scoring) | Mathematical problem and step-by-step reasoning steps (Q, s_1...s_i) | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Analysis of the step + correctness judgment ("Yes"/"No") + scalar reward score | 1D (t); 0D | Not specified in the paper. |

## Summary
The paper focuses on process-level evaluation of mathematical reasoning steps, where a reward model assesses step-by-step solutions. Inputs are textual problem statements and ordered reasoning steps, and outputs include natural-language analyses, correctness judgments, and a scalar reward score. The explicit evidence supports 1D (t) text outputs and a scalar reward, while interface dynamics, attention dynamics, and state dynamics are not specified.

## Evidence
### Task: Process-level step evaluation (analysis + correctness judgment / reward scoring)
- "Given a mathematical problem Q, the policy model generates a sequential chain-of-reasoning process S = {s_1, s_2, ..., s_n}" (Section 3.1 Reasoning for Process Reward Modeling)
- "First, G generates a comprehensive analysis A_i of each reasoning step s_i" (Section 3.1 Reasoning for Process Reward Modeling)
- "Then, G generates a natural language judgment J_i indicating the correctness of the step" (Section 3.1 Reasoning for Process Reward Modeling)
- "y_j denotes the j-th token in the output sequence Y_i" (Section 3.1 Reasoning for Process Reward Modeling)
- "we calculate the average probability of \"Yes\" judgments (using softmax with \"No\" judgments) as the reward" (Section 3.3 Inference-Time Scaling Strategy)
