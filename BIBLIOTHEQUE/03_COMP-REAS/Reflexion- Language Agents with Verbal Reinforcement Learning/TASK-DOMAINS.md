# Reflexion: Language Agents with Verbal Reinforcement Learning (2023)
Source: Reflexion- Language Agents with Verbal Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sequential decision-making / control in interactive environments | Text observations, task instructions, and trajectory history (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Text actions / action trajectories (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Search-based question answering with retrieval and reasoning | Question plus retrieved Wikipedia context (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Answer text (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Context-grounded reasoning question answering | Question plus provided ground-truth context (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Answer text (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Code generation / program synthesis | Natural-language problem description, prior implementation, unit-test feedback, and self-reflection text (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Function body / source code text (tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers four text-centric task settings: sequential decision-making/control, search-based question answering, context-grounded reasoning QA, and code generation/program synthesis. All supported tasks operate over token sequences, so the justified domain is 1D (t), and interface limits are consistently Capped due explicit limits such as action thresholds, memory-size bounds, retry bounds, and unit-test caps. Attention is Dynamic when agents choose actions/retrieval or iterate via test feedback, and Static in the CoT reasoning-only setup with fixed provided context. State is Constructed across tasks because Reflexion stores and reuses self-reflective memory between trials.

## Evidence
### Task: Sequential decision-making / control in interactive environments
- "We perform experiments on (1) decision-making tasks to test sequential action choices over long trajectories" (Section 1 Introduction)
- "AlfWorld is a suite of text-based environments that challenge an agent to solve multi-step tasks in a variety of interactive environments" (Section 4.1 Sequential decision making: ALFWorld)
- "if the number of actions taken in the current environment exceeds 30 (inefficient planning), we self-reflect." (Section 4.1 Sequential decision making: ALFWorld)
- "WebShop is a web-based problem-solving benchmark that tests agents to navigate an e-commerce website to locate and purchase products given requests from clients." (Section B.1 WebShop Limitation)
- Inference: Input/output were mapped to token trajectories and actions from the paper's text-based environment/action descriptions; 1D (t) and Dynamic attention follow from sequential interaction and action choice; Capped dynamics follows from explicit action and memory bounds ("exceeds 30" actions; "truncate the agent's memory to the last 3 self-reflections"). State is Constructed because Reflexion stores persistent memory for later trials.

### Task: Search-based question answering with retrieval and reasoning
- "To test holistic question and answering ability, which requires reasoning and action choice, we implement a Reflexion + ReAct [30] agent that can retrieve relevant context using a Wikipedia API and infer answers using step-by-step explicit thinking." (Section 4.2 Reasoning: HotpotQA)
- "HotPotQA [28] is a Wikipedia-based dataset with 113k question-and-answer pairs that challenge agents to parse content and reason over several supporting documents." (Section 4.2 Reasoning: HotpotQA)
- "In the Reflexion runs, we allowed the agent to gather experience and retry on failed tasks until it produced 3 consecutive failed attempts on the particular task." (Section 4.2 Reasoning: HotpotQA, Results)
- Inference: The address space is 1D (t) because questions, retrieved context, and answers are textual token streams. Attention is Dynamic due runtime retrieval/action choice. Dynamics is Capped from explicit retry and memory bounds, and State is Constructed from the self-reflection memory loop.

### Task: Context-grounded reasoning question answering
- "To test improvement in reasoning only ability, we implement Reflexion + Chain-of-Thought (CoT) [26] for step-by-step  $Q \\to A$  and Q,  $C_{gt} \\to A$  implementations, where Q is the question,  $C_{gt}$  is the ground truth context from the dataset, and A is the final answer." (Section 4.2 Reasoning: HotpotQA)
- "Since CoT is not a multi-step decision-making technique, we give  $C_{gt}$  to the agent so that we can isolate the reasoning behavior over large sections of the provided text." (Section 4.2 Reasoning: HotpotQA)
- "After each trial, the self-reflection loop is employed to amplify the binary signal, similar to the decision-making setup 4.1 in AlfWorld with a memory size of 3 experiences." (Section 4.2 Reasoning: HotpotQA)
- Inference: 1D (t) follows from text question/context/answer IO. Attention is Static because this setup uses provided context rather than runtime retrieval/actions. Dynamics is Capped from context-window and explicit memory limits, and State is Constructed because prior reflective summaries are reused between trials.

### Task: Code generation / program synthesis
- "We evaluate the baseline and Reflexion approaches on Python and Rust code writing on MBPP [2], HumanEval [6], and LeetcodeHardGym, our new dataset." (Section 4.3 Programming)
- "MBPP and HumanEval measure function body generation accuracy given natural language descriptions." (Section 4.3 Programming)
- "To generate a test suite, we use Chain-of-Thought prompting [26] to produce diverse, extensive tests" (Section 4.3 Programming)
- "We set n to a maximum of 6 unit tests." (Section 4.3 Programming)
- "the setup for the learning loop for a Reflexion programming agent is identical to the reasoning and decision-making agents with a max memory limit of 1 experience." (Section 4.3 Programming)
- Inference: Input/output are token sequences (natural-language specs, code, test feedback, self-reflections), so dimension is 1D (t). Dynamics is Capped from explicit unit-test and memory limits. Attention is Dynamic because the loop adapts edits from runtime execution/test feedback, and State is Constructed via persistent reflective memory.
