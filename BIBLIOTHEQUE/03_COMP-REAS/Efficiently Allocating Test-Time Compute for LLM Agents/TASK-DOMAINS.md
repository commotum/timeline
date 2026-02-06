# Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents (Not specified in the paper)
Source: Efficiently Allocating Test-Time Compute for LLM Agents.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Graph search navigation in POGS (sequential decision-making/control) | Natural language observation + interaction history + prior plan | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Natural language action command (single node choice) + optional plan | 1D (t) (inferred) | Capped (inferred) |
| Survival/resource management/crafting in Crafter (sequential decision-making/control) | Natural language observation + interaction history + prior plan | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Natural language action command + optional plan | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates LLM agents on two sequential decision-making tasks: graph search navigation in POGS and long-horizon survival/crafting in Crafter, both mediated through natural language observations and action outputs. Inputs and outputs are text sequences, while the agent’s context is explicitly truncated to a fixed history length, and outputs are constrained to an optional plan plus a single action. Across both tasks, planning is an explicit, reusable artifact in the interaction history, supporting constructed state, with attention effectively tied to a fixed chat-formatted input slice.

## Evidence
### Task: Graph search navigation in POGS (sequential decision-making/control)
- "Agents navigate procedurally generated graph using only local observations" (Section 4.1)
- "must find a path to a target node." (Appendix B.2)
- "Interaction in both environments occurs via natural language." (Section 4.1)
- "Your action should be a single integer representing the label of the node you want to travel to." (Appendix B.2, Prompt 21)
- "At each timestep t, the agent receives its history and current observation  $o_t$  within a chat template" (Section 4.2)
- "If they choose to plan, they output the plan followed by the action, using the format <plan> [natural language plan] </plan> [Action]." (Section 4.2)
- "using its content as the current plan  $p_t$  in subsequent context." (Section 4.2)
- "Output nothing else except an optional <plan>...</plan> block and that single action." (Appendix A.3, Prompt 20)
- "history provided to the agent was truncated to a maximum of 16 observations." (Figure 4)
- Inference: Labeled input/output as 1D (t), dynamics as Capped, attention as Static, and state as Constructed because observations/actions are natural language, history is truncated to 16 observations, input is a fixed chat history, and plans are reused in subsequent context (see quotes above).

### Task: Survival/resource management/crafting in Crafter (sequential decision-making/control)
- "Second, **Crafter** (Hafner, 2022) is a complex 2D grid-world" (Section 4.1)
- "It demands multi-scale planning for survival, resource management, and crafting" (Section 4.1)
- "Crafter presents a procedurally generated 2D world where the agent must gather resources, craft tools, and defend against creatures to survive" (Appendix B.1)
- "This prompt is then passed to the LLM, which processes the contextual information and generates the subsequent action as a natural language string." (Appendix B.1)
- "At each timestep t, the agent receives its history and current observation  $o_t$  within a chat template" (Section 4.2)
- "If they choose to plan, they output the plan followed by the action, using the format <plan> [natural language plan] </plan> [Action]." (Section 4.2)
- "using its content as the current plan  $p_t$  in subsequent context." (Section 4.2)
- "Output nothing else except an optional <plan>...</plan> block and that single action." (Appendix A.3, Prompt 20)
- "history provided to the agent was truncated to a maximum of 16 observations." (Figure 4)
- Inference: Labeled input/output as 1D (t), dynamics as Capped, attention as Static, and state as Constructed because observations/actions are natural language, history is truncated to 16 observations, input is a fixed chat history, and plans are reused in subsequent context (see quotes above).
