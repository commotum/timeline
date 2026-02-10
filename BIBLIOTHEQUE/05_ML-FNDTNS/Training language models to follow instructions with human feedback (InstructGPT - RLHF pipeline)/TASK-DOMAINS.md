# Training language models to follow instructions with human feedback (2022)
Source: Training language models to follow instructions with human feedback (InstructGPT - RLHF pipeline).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Instruction-following text generation | Natural-language prompts/instructions | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Generated natural-language completions | 1D (t) (inferred) | Capped (inferred) |
| Open-domain question answering | Natural-language questions/prompts | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer text | 1D (t) (inferred) | Capped (inferred) |
| Closed-domain question answering | Question plus provided source text/context | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Grounded answer text | 1D (t) (inferred) | Capped (inferred) |
| Brainstorming / ideation | Idea-generation prompt text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | List of ideas in text | 1D (t) (inferred) | Capped (inferred) |
| Dialogue / chat response generation | Multi-turn chat history/prompt text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Assistant response text | 1D (t) (inferred) | Capped (inferred) |
| Rewrite / transformation | Source text plus rewrite instruction | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Rewritten/transformed text | 1D (t) (inferred) | Capped (inferred) |
| Summarization | Long-form text (article/transcript/post) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Summary text | 1D (t) (inferred) | Capped (inferred) |
| Information extraction | Source text/documents | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Extracted items/entities as text | 1D (t) (inferred) | Capped (inferred) |
| Classification | Text snippet/prompt | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Label/class choice | 0D (inferred) | Fixed (inferred) |
| Machine translation (Fr -> En) | Source-language text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Target-language translated text | 1D (t) (inferred) | Capped (inferred) |
| Code understanding (code QA/summarization/description) | Code snippets and/or code-related questions in text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Code explanations/summaries/answers in text | 1D (t) (inferred) | Capped (inferred) |
| Preference scoring/ranking (reward modeling) | Prompt and candidate response(s) text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Scalar reward / preference score | 0D (inferred) | Fixed (inferred) |
| Other natural-language tasks | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper covers a broad set of text-centric instruction-following tasks, with explicit API categories spanning generation, QA, brainstorming, chat, rewrite, summarization, classification, and extraction, plus additional translation and code tasks. The reward model adds a separate scalar preference-scoring task over prompt-response pairs. Most tasks are justified as token-sequence input/output domains (1D (t)), with capped dynamics inferred from the stated token limits. Attention and state are inferred as static and direct from the GPT-3 prompt-to-completion setup described in the OCR.

## Evidence
### Task: Instruction-following text generation
- "we use reinforcement learning from human feedback (RLHF; Christiano et al., 2017; Stiennon et al., 2020) to fine-tune GPT-3 to follow a broad class of written instructions" (Section 1 Introduction)
- "| Generation     | 45.6% |" (Table 1)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from "All model architectures use the GPT-3 architecture" and "All our language models and RL policies have a context length of 2k tokens. We filter out prompts that are longer than 1k tokens and limit the maximum response length to 1k tokens." (Appendix C)

### Task: Open-domain question answering
- "These prompts are very diverse and include generation, question answering, dialog, summarization, extractions, and other natural language tasks (see Table 1)." (Section 3.3 Tasks)
- "| Open QA        | 12.4% |" (Table 1)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Closed-domain question answering
- "| Closed QA      | 2.6%  |" (Table 1)
- "On \"closed-domain\" tasks from our API prompt distribution, where the output should not contain information that is not present in the input (e.g. summarization and closed-domain QA)" (Abstract)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Brainstorming / ideation
- "| Brainstorming  | 11.2% |" (Table 1)
- "| Brainstorming | List five ideas for how to regain enthusiasm for my career" (Table 2)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Dialogue / chat response generation
- "These prompts are very diverse and include generation, question answering, dialog, summarization, extractions, and other natural language tasks (see Table 1)." (Section 3.3 Tasks)
- "| Chat           | 8.4%  |" (Table 1)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Rewrite / transformation
- "| Rewrite        | 6.6%  |" (Table 1)
- "| Rewrite       | This is the summary of a Broadway play:" (Table 2)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Summarization
- "These prompts are very diverse and include generation, question answering, dialog, summarization, extractions, and other natural language tasks (see Table 1)." (Section 3.3 Tasks)
- "| Summarization  | 4.2%  |" (Table 1)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Information extraction
- "These prompts are very diverse and include generation, question answering, dialog, summarization, extractions, and other natural language tasks (see Table 1)." (Section 3.3 Tasks)
- "| Extract        | 1.9%  |" (Table 1)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Classification
- "| Classification | 3.5%  |" (Table 1)
- "For each snippet of text, label the sentiment of the text as positive or negative." (Appendix D.2, SST)
- Inference: In/Out token handling, capped dynamics, and static/direct attention/state are inferred from Appendix C GPT-3/token-limit details; output as 0D with Fixed dynamics is inferred from the explicit label-choice framing in Appendix D.2 (e.g., "Label: [positive / negative]").

### Task: Machine translation (Fr -> En)
- "we observe performance regressions compared to GPT-3 on certain public NLP datasets, notably ... WMT 2015 French to English translation" (Section 1 Introduction)
- "Translate the following sentences from French into English." (Appendix D.2, WMT Fr -> En 15)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Code understanding (code QA/summarization/description)
- "it is able to follow instructions for summarizing code, answer questions about code" (Section 1 Introduction)
- "we find that InstructGPT shows ability to follow instructions in non-English languages, and perform summarization and question-answering for code." (Section 4.3 Qualitative results)
- Inference: In/Out Dimension as 1D (t), In/Out Dynamics as Capped, and Attention/State as Static/Direct are inferred from Appendix C token-length limits and GPT-3 architecture.

### Task: Preference scoring/ranking (reward modeling)
- "we collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback." (Abstract)
- "we trained a model to take in a prompt and response, and output a scalar reward." (Section 3.5 Models)
- Inference: Input as 1D (t), capped input dynamics, and static/direct attention/state are inferred from Appendix C GPT-3/token-limit details; output as 0D with Fixed dynamics is inferred from the explicit "scalar reward" output.

### Task: Other natural-language tasks
- "These prompts are very diverse and include generation, question answering, dialog, summarization, extractions, and other natural language tasks (see Table 1)." (Section 3.3 Tasks)
- "| Other          | 3.5%  |" (Table 1)
