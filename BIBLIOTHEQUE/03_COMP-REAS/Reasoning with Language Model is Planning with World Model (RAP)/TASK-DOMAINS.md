# Reasoning with Language Model is Planning with World Model (Not specified in the paper)
Source: Reasoning with Language Model is Planning with World Model (RAP).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plan generation (Blocksworld) | Block configurations and goals (natural language text) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed | Action plan (sequence of actions) | 1D (t) (inferred) | Capped (inferred) |
| Math reasoning (GSM8K) | Problem description and final question (text) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed | Final answer | 0D (inferred) | Fixed (inferred) |
| Logical reasoning / hypothesis verification (PrOntoQA) | Facts, logical rules, and hypothesis (text) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed | True/false answer with proof | 0D; 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper applies RAP to plan generation, math word problem solving, and logical hypothesis verification, all framed as text-based problems. Inputs are 1D (t) language sequences (inferred), while outputs range from single final answers to action/proof sequences; plan and proof lengths are capped by step/depth limits, and math outputs are single answers (fixed, inferred). Across tasks, the system constructs explicit intermediate states (block configurations, intermediate variables, or focused facts) and uses dynamic, search-based attention over reasoning steps (inferred from MCTS).

## Evidence
### Task: Plan generation (Blocksworld)
- "The plan generation task aims to produce a sequence of actions to achieve a given goal" (Section 4.1 Plan Generation)
- "it is natural to define a state as the configuration of blocks (described in natural language)" (Section 3.1 Language Model as World Model)
- "There are at most 5 blocks in each test case." (Section 4.1 Plan Generation)
- "Once a state has met all conditions in the goal or the depth limit of the tree is reached" (Section 4.1 Plan Generation)
- Inference: Labeled dimensions as 1D (t) and attention as Dynamic because states/actions are "described in natural language" and RAP "strategically builds a reasoning tree by iteratively considering the most promising reasoning steps" (Introduction). Capped dynamics inferred from "There are at most 5 blocks in each test case" and the "depth limit of the tree" (Section 4.1).

### Task: Math reasoning (GSM8K)
- "Math reasoning tasks, such as GSM8k (Cobbe et al., 2021), often include a description and a final question." (Section 4.2 Math Reasoning)
- "We define a **state** as the values of intermediate variables" (Section 4.2 Math Reasoning)
- "only the final answer is required" (Section 3.4 RAP-Aggregation)
- Inference: Labeled input as 1D (t) because the task uses a "description and a final question" in text (Section 4.2). Labeled attention as Dynamic from "strategically builds a reasoning tree by iteratively considering the most promising reasoning steps" (Introduction). Labeled output as 0D and Fixed because "only the final answer is required" (Section 3.4).

### Task: Logical reasoning / hypothesis verification (PrOntoQA)
- "A logical reasoning task (e.g. PrOntoQA (Saparov and He, 2022)) typically provides a set of facts and logical rules" (Section 4.3 Logical Reasoning)
- "a model is required to verify if a hypothesis fact is true or false" (Section 4.3 Logical Reasoning)
- "These tasks not only require the correct final answer (true/false), but also a detailed proof" (Section 4.3 Logical Reasoning)
- "we define the state as a fact we are focusing on" (Section 4.3 Logical Reasoning)
- "examples requiring 3, 4, and 5 reasoning hops in a correct proof" (Section 4.3 Logical Reasoning)
- Inference: Labeled input as 1D (t) because facts/rules/hypothesis are provided in text; labeled attention as Dynamic from "strategically builds a reasoning tree by iteratively considering the most promising reasoning steps" (Introduction); labeled output dimension as 0D; 1D (t) because the task needs a true/false answer plus a proof; labeled output dynamics as Capped based on "3, 4, and 5 reasoning hops" (Section 4.3).
