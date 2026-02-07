# HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models (Year not specified)
Source: HoPE- Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (next-token prediction) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Question answering | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Natural language inference | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Summarization | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates HoPE on text-only tasks spanning language modeling (perplexity/next-token prediction) and SCROLLS downstream tasks in question answering, natural language inference, and summarization. Inputs and outputs are token sequences (inferred), and the experiments use fixed sequence lengths (1024 pre-training; 8192 fine-tuning), so the task interfaces are 1D (t) and capped (inferred). Attention is treated as static and state as direct based on the standard decoder-only Transformer and next-token prediction objective (inferred).

## Evidence
### Task: Language modeling (next-token prediction)
- "The next token prediction objective is adopted for language model training." (Section 8.3.1 Perplexity Experiment)
- "evaluate the log perplexity of pre-trained language models" (Section 4.2 Perplexity Experiment (PPL))
- "Let  $\mathbb{S}_N = \{w_i\}_{i=1}^N$  be a sequence of N input tokens" (Section 2.1 Relative position encoding)
- "The pre-training sequence length is set to 1024" (Section 4.2 Perplexity Experiment (PPL))
- "We choose the standard decoder-only Transformer(Touvron et al., 2023) as the base model" (Section 4.2 Perplexity Experiment (PPL))
- Inference: Mapped input/output to 1D (t) token sequences with capped dynamics and static/direct processing based on "sequence of N input tokens," the fixed sequence length, the "standard decoder-only Transformer," and the "next token prediction objective." (Sections 2.1, 4.2, 8.3.1)

### Task: Question answering
- "Question-Answering (Qasper(Dasigi et al., 2021), NarrativeQA(Kočiský et al., 2017), and QuALITY(Pang et al., 2022))" (Section 8.3.2 Fine-Tuning Experiment)
- "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment)
- "Let  $\mathbb{S}_N = \{w_i\}_{i=1}^N$  be a sequence of N input tokens" (Section 2.1 Relative position encoding)
- Inference: Classified input/output as 1D (t) token sequences with capped dynamics and static/direct processing based on the token-sequence definition, the fixed 8192 sequence length, and the next-token objective. (Sections 2.1, 8.3.2)

### Task: Natural language inference
- "Natural Language Inference (ContractNLI(Koreeda and Manning, 2021))" (Section 8.3.2 Fine-Tuning Experiment)
- "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment)
- "Let  $\mathbb{S}_N = \{w_i\}_{i=1}^N$  be a sequence of N input tokens" (Section 2.1 Relative position encoding)
- Inference: Classified input/output as 1D (t) token sequences with capped dynamics and static/direct processing based on the token-sequence definition, the fixed 8192 sequence length, and the next-token objective. (Sections 2.1, 8.3.2)

### Task: Summarization
- "Summarization (QMSum(Zhong et al., 2021), SummScreenFD(Chen et al., 2022), and GovReport(Huang et al., 2021))." (Section 8.3.2 Fine-Tuning Experiment)
- "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment)
- "Let  $\mathbb{S}_N = \{w_i\}_{i=1}^N$  be a sequence of N input tokens" (Section 2.1 Relative position encoding)
- Inference: Classified input/output as 1D (t) token sequences with capped dynamics and static/direct processing based on the token-sequence definition, the fixed 8192 sequence length, and the next-token objective. (Sections 2.1, 8.3.2)
