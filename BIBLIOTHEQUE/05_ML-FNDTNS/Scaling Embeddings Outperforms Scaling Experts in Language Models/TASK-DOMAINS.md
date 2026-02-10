# Scaling Embeddings Outperforms Scaling Experts in Language Models (Not specified in the paper)
Source: Scaling Embeddings Outperforms Scaling Experts in Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoregressive language modeling / token prediction | Token sequences | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token outputs (tokens) (inferred) | 1D (t) | Capped (inferred) |
| General-domain language understanding QA | Question/prompt tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer tokens | 1D (t) | Capped (inferred) |
| Reasoning QA | Reasoning question tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Reasoned answer tokens | 1D (t) | Capped (inferred) |
| Code generation | Coding problem and instruction tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Source code tokens | 1D (t) | Capped (inferred) |
| Agentic tool-use workflow execution | Multi-turn instruction and tool-feedback tokens (inferred) | 1D (t) | Capped (inferred) | Dynamic (inferred) | Direct (inferred) | Tool-call and response tokens (inferred) | 1D (t) | Capped (inferred) |
| Agentic coding / software engineering execution | Software issue descriptions, repository context, and terminal feedback tokens (inferred) | 1D (t) | Capped (inferred) | Dynamic (inferred) | Direct (inferred) | Code-edit and terminal-command tokens (inferred) | 1D (t) | Capped (inferred) |
| Mathematical reasoning problem solving | Math problem tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Mathematical solution tokens | 1D (t) | Capped (inferred) |

## Summary
The paper covers language-model pretraining and chat/base downstream evaluation across general QA, reasoning QA, coding, agentic tool use, agentic coding, and mathematical reasoning. The OCR supports a single core modality of token streams, so all tasks are classified as 1D (t) input/output. Dynamics are capped (inferred) from explicit sequence limits up to 256k tokens, and state is direct (inferred) because no persistent constructed internal-state mechanism is specified. Attention is static (inferred) for standard text tasks, with dynamic attention inferred only for agentic rows due explicit tool-integration workflow execution.

## Evidence
### Task: Autoregressive language modeling / token prediction
- "All models are pre-trained on a corpus of 300B tokens." (Section 3, Experiment Settings)
- "It is first pre-trained on 11T tokens with a sequence length of 8k, followed by 1.5T tokens of mid-training during which the sequence length is extended to 128k, and is finally trained on SFT data. To support extended context, we implement YARN [Peng et al., 2023] during the 32k sequence length training stage, enabling LongCat-Flash-Lite to handle sequences up to 256k tokens." (Section 6.1, Training Data)
- "projection directly to the N-gram Embedding outputs, we are investigating a broader design space to fully exploit the captured local context for efficient token prediction." (Section 4.3)
- Inference: 1D token input/output, capped dynamics, static attention, and direct state are inferred from the token-sequence training/evaluation setup and explicit sequence-length caps; no constructed-state mechanism is described.

### Task: General-domain language understanding QA
- "- General Tasks: MMLU [Hendrycks et al., 2021], MMLU-Pro [Wang et al., 2024], C-Eval [Huang et al., 2023], and CMMLU [Li et al., 2023]." (Section 6.2)
- "General Domains. LongCat-Flash-Lite delivers balanced and competitive performance in general domain knowledge tasks." (Section 6.3)
- Inference: input/output token streams and 1D (t) indexing are inferred from the paper’s language-model setup; capped dynamics are supported by the 256k-token sequence limit in Section 6.1; static attention and direct state are inferred because no dynamic retrieval controller or constructed internal state is specified.

### Task: Reasoning QA
- "- Reasoning Tasks: BBH [Suzgun et al., 2023], GPQA [M-A-P Team, ByteDance., 2025], DROP [Dua et al., 2019] and GSM8K [Cobbe et al., 2021]." (Section 6.2)
- "To assess downstream performance, we evaluate both models on benchmarks spanning three core capability domains:" (Section 6.2)
- Inference: reasoning prompts and answers are treated as token streams (1D (t)); capped dynamics, static attention, and direct state follow the same capped-context LM interface described in Section 6.1.

### Task: Code generation
- "- Coding Tasks: HumanEval+ [Liu et al., 2024], MultiPL-E [Cassano et al., 2022], and BigCodeBench [Zhuo et al., 2025]." (Section 6.2)
- "Agentic Coding. In coding-related tasks, LongCat-Flash-Lite demonstrates remarkable practical problem-solving capabilities." (Section 6.3)
- Inference: coding inputs/outputs are represented as tokenized text/code sequences (1D (t)); capped dynamics are inferred from sequence-length limits; static attention and direct state are inferred because constructed internal coding state is not explicitly described.

### Task: Agentic tool-use workflow execution
- "- Agentic Tool Use Tasks:  $\tau^2$  Bench [Barres et al., 2025], Vita Bench [He et al., 2025]." (Section 6.3)
- "This leading score underscores LongCat-Flash-Lite's superior ability to handle complex, real-world task workflows via tool integration in practical business scenarios." (Section 6.3)
- Inference: multi-turn instruction/tool-feedback token streams and token/action outputs are inferred from the tool-use benchmark framing; dynamic attention is inferred from explicit tool-integration workflow execution; capped dynamics are inferred from the model’s sequence-length limits in Section 6.1; state remains direct because no explicit constructed internal state is specified.

### Task: Agentic coding / software engineering execution
- "- Agentic Coding Tasks: SWE-Bench [Jimenez et al., 2023], TerminalBench [Merrill et al., 2026], SWE-Bench Multiligual [Yang et al., 2025], and PRDBench [Fu et al., 2025]." (Section 6.3)
- "In TerminalBench, which evaluates terminal command execution competence, LongCat-Flash-Lite secures a leading score of 33.75, far exceeding Qwen3-Next-80B-A3B-Instruct (15.19), Gemini 2.5 Flash-Lite (20.0) and Kimi-Linear-48B-A3B (20.0), reflecting its robust ability to understand and execute terminal-related instructions critical for developer-centric agentic applications." (Section 6.3)
- "We observe that our model can autonomously write unit tests to verify its development, producing higher-quality code repositories." (Section 6.3)
- Inference: software-issue/repository/terminal context and generated edits/commands are represented as token streams (1D (t)); dynamic attention is inferred from terminal/tool execution behavior; capped dynamics follow Section 6.1 sequence limits; direct state is inferred because constructed internal state is not explicitly described.

### Task: Mathematical reasoning problem solving
- "- Mathematical Reasoning Tasks: MATH500 [Lightman et al., 2023], AIME24 [MAA, 2024], AIME25 [MAA, 2025]." (Section 6.3)
- "**Mathematical Reasoning.** LongCat-Flash-Lite exhibits strong mathematical reasoning capabilities across both basic and advanced tasks." (Section 6.3)
- "59.58) and Gemini 2.5 Flash-Lite (63.33 and 50.1), highlighting its ability to handle complex, multi-step mathematical deduction." (Section 6.3)
- Inference: math problems and solutions are handled as token sequences (1D (t)); capped dynamics derive from explicit sequence-length bounds in Section 6.1; static attention and direct state are inferred since no dynamic retrieval controller or constructed internal state is described.
