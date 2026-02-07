# REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS (Not specified in the paper.)
Source: ReAct- Synergizing Reasoning and Acting in Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Question answering (multi-hop, HotpotQA) | question (text) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | answer (text; via finish[answer]) | 1D (t) (inferred) | Capped (inferred) |
| Fact verification (FEVER) | claim (text) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | label: SUPPORTS / REFUTES / NOT ENOUGH INFO | 0D (inferred) | Capped (inferred) |
| Interactive decision making in text-based game (ALFWorld) | high-level goal + text observations | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | text actions (e.g., go to..., take..., use...) | 1D (t) (inferred) | Open (inferred) |
| Webpage navigation / online shopping (WebShop) | user instruction + webpage text (product titles, descriptions, options) | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | web interaction actions (search, choose product, choose options, buy) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper evaluates ReAct on four language-centric tasks: multi-hop question answering, fact verification, a text-based household game, and web shopping/navigation. Inputs and outputs are primarily textual sequences (1D (t)), with FEVER producing categorical labels (0D). For HotpotQA and FEVER, the interaction is explicitly capped by step limits, while ALFWorld and WebShop are described as long-horizon interactive environments, supporting open-ended multi-step interaction; attention and state are inferred as dynamic/constructed due to action selection and reasoning traces.

## Evidence
### Task: Question answering (multi-hop, HotpotQA)
- "Hot-PotQA (Yang et al., 2018), a multi-hop question answering benchmark that requires reasoning over two or more Wikipedia passages" (Section 3.1 Setup).
- "models only receive the question/claim as input without access to support paragraphs" (Section 3.1 Setup).
- "finish[answer], which would finish the current task with answer." (Section 3.1 Setup, Action Space).
- Inference: Marked `1D (t)` for input/output because the task uses a "question" and an "answer"; marked `Dynamic` attention because ReAct works "by interacting with a Wikipedia API"; marked `Constructed` state because thoughts "update the context  c_{t+1} = (c_t, \hat{a}_t)"; marked `Capped` dynamics because "We set 7 and 5 steps for HotpotQA and FEVER respectively." (Sections 3, 2, 3.2).

### Task: Fact verification (FEVER)
- "FEVER (Thorne et al., 2018), a fact verification benchmark where each claim is annotated SUPPORTS, REFUTES, or NOT ENOUGH INFO" (Section 3.1 Setup).
- "models only receive the question/claim as input without access to support paragraphs" (Section 3.1 Setup).
- Inference: Marked `1D (t)` for input because the task input is a natural-language "claim"; marked `0D` output because the claim is annotated with categorical labels; marked `Dynamic` attention because ReAct retrieves information "by interacting with a Wikipedia API"; marked `Constructed` state because thoughts "update the context  c_{t+1} = (c_t, \hat{a}_t)"; marked `Capped` dynamics because "We set 7 and 5 steps for HotpotQA and FEVER respectively." (Sections 3.1, 3, 2, 3.2).

### Task: Interactive decision making in text-based game (ALFWorld)
- "includes 6 types of tasks in which an agent needs to achieve a high-level goal (e.g. examine paper under desklamp)" (Section 4 Decision Making Tasks).
- "by navigating and interacting with a simulated household via text actions (e.g., go to coffeetable 1, take paper 2, use desklamp 1)." (Section 4 Decision Making Tasks).
- "A task instance can have more than 50 locations and take an expert policy more than 50 steps to solve" (Section 4 Decision Making Tasks).
- "each trajectory includes sparse thoughts that (1) decompose the goal, (2) track subgoal completion" (Section 4 Decision Making Tasks).
- Inference: Marked `1D (t)` for input/output because goals, observations, and actions are text; marked `Open` dynamics because instances can take "more than 50 steps" and involve long-horizon interaction; marked `Dynamic` attention because the agent is "navigating and interacting" via actions; marked `Constructed` state because thoughts "decompose the goal" and "track subgoal completion." (Section 4 Decision Making Tasks).

### Task: Webpage navigation / online shopping (WebShop)
- "requires an agent to purchase a product based on a user instruction" (Section 4 Decision Making Tasks).
- "contains a high variety of structured and unstructured texts (e.g. product titles, descriptions, and options crawled from Amazon)" (Section 4 Decision Making Tasks).
- "actions to search, choose product, choose options, and buy" (Section 4 Decision Making Tasks).
- "ReAct prompts additionally reasoning to determine what to explore, when to buy, and what products options are relevant to the instruction." (Section 4 Decision Making Tasks).
- Inference: Marked `1D (t)` for input/output because instructions, webpage text, and actions are language; marked `Open` dynamics because the task proceeds through multi-step web interactions; marked `Dynamic` attention because the agent selects searches/options via actions; marked `Constructed` state because ReAct reasons "to determine what to explore, when to buy." (Section 4 Decision Making Tasks).
