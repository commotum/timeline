# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (Not specified in the paper)
Source: DeepSeek-R1- Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathematical problem solving | text prompts/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers/solutions | 1D (t) (inferred) | Not specified in the paper. |
| Coding competition problem solving | text prompts/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text responses | 1D (t) (inferred) | Not specified in the paper. |
| Software engineering coding tasks | text prompts/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text responses | 1D (t) (inferred) | Not specified in the paper. |
| Knowledge question answering (educational) | text prompts/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Factual question answering | text prompts/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Long-context document QA/analysis | text prompts/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers/analysis | 1D (t) (inferred) | Not specified in the paper. |
| Instruction-following (format adherence) | text prompts/instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | formatted text outputs | 1D (t) (inferred) | Not specified in the paper. |
| Creative writing | text prompts/instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | generated text | 1D (t) (inferred) | Not specified in the paper. |
| General question answering (open-domain) | text prompts/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Editing | text prompts/instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | edited text | 1D (t) (inferred) | Not specified in the paper. |
| Summarization | text prompts/instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | summaries (text) | 1D (t) (inferred) | Not specified in the paper. |
| Function calling | text prompts/instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | function-call text | 1D (t) (inferred) | Not specified in the paper. |
| Multi-turn dialogue | text prompts/conversation history | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | dialogue responses (text) | 1D (t) (inferred) | Not specified in the paper. |
| Role-playing | text prompts/instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | role-play responses (text) | 1D (t) (inferred) | Not specified in the paper. |
| JSON structured output generation | text prompts/instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | JSON text output | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper frames DeepSeek-R1 as a text-based reasoning model covering math, coding/engineering tasks, knowledge and factual QA, long-context document QA, instruction-following, and general language tasks such as writing, editing, summarization, and open-domain QA. It also discusses structured interaction tasks (function calling, JSON output) and multi-turn/role-playing settings. Inputs and outputs are text sequences (1D (t) inferred), while interface dynamics, attention policy, and state construction are not specified.

## Evidence
### Task: Mathematical problem solving
- "On math tasks, DeepSeek-R1 demonstrates performance on par with OpenAI-o1-1217." (Section 3.1 DeepSeek-R1 Evaluation)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Coding competition problem solving
- "On coding-related tasks, DeepSeek-R1 demonstrates expert level in code competition tasks." (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Software engineering coding tasks
- "For engineering-related tasks, DeepSeek-R1 performs slightly better than DeepSeek-V3." (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Knowledge question answering (educational)
- "On benchmarks such as MMLU, MMLU-Pro, and GPQA Diamond" (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Factual question answering
- "On the factual benchmark SimpleQA, DeepSeek-R1 outperforms DeepSeek-V3, demonstrating its capability in handling fact-based queries." (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Long-context document QA/analysis
- "FRAMES, a long-context-dependent QA task, showcasing its strong document analysis capabilities." (Section 3.1 DeepSeek-R1 Evaluation)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Instruction-following (format adherence)
- "IF-Eval, a benchmark designed to assess a model's ability to follow format instructions." (Section 3.1 DeepSeek-R1 Evaluation)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Creative writing
- "including creative writing, general question answering, editing, summarization, and more." (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: General question answering (open-domain)
- "including creative writing, general question answering, editing, summarization, and more." (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Editing
- "including creative writing, general question answering, editing, summarization, and more." (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Summarization
- "including creative writing, general question answering, editing, summarization, and more." (Section 1.2 Summary of Evaluation Results)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Function calling
- "tasks such as function calling, multi-turn, complex role-playing, and JSON output." (Section 5 Conclusion, Limitations, and Future Work)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Multi-turn dialogue
- "tasks such as function calling, multi-turn, complex role-playing, and JSON output." (Section 5 Conclusion, Limitations, and Future Work)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: Role-playing
- "tasks such as function calling, multi-turn, complex role-playing, and JSON output." (Section 5 Conclusion, Limitations, and Future Work)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)

### Task: JSON structured output generation
- "tasks such as function calling, multi-turn, complex role-playing, and JSON output." (Section 5 Conclusion, Limitations, and Future Work)
- Inference: In/Out Dimension set to 1D (t) because the template defines a prompt/assistant text exchange: "User: prompt. Assistant:" (Table 1 | Template for DeepSeek-R1-Zero)
