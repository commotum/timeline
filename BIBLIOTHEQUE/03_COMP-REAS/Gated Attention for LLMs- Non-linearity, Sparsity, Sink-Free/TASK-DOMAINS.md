# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free (Not specified in the paper.)
Source: Gated Attention for LLMs- Non-linearity, Sparsity, Sink-Free.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (perplexity) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| English benchmark evaluation (Hellaswag) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| General knowledge benchmark evaluation (MMLU) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Math reasoning benchmark evaluation (GSM8k) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Coding benchmark evaluation (HumanEval) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Chinese proficiency benchmark evaluation (C-eval, CMMLU) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Long-context benchmark evaluation (RULER) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates gated-attention LLMs on language modeling perplexity and several few-shot benchmarks spanning English, general knowledge, math reasoning, coding, and Chinese proficiency, plus long-context evaluation on RULER. Across these tasks, inputs and outputs are treated as token sequences with 1D (t) structure and capped dynamics tied to stated context windows (inferred). Attention is classified as Static and state as Direct based on the fixed-context softmax attention formulation described in the model (inferred).

## Evidence
### Task: Language modeling (perplexity)
- "We also report the perplexity (PPL) of language modeling" (Section 3.1, Evaluation)
- "We train the models on subsets of a 3.5T high-quality tokens" (Section 3.1, Model Architecture and Training Settings)
- "The context sequence length is set to 4096." (Section 3.1, Model Architecture and Training Settings)
- "Given an input  $X \in \mathbb{R}^{n \times d_{\text{model}}}$ , where n is the sequence length" (Section 2.1, Preliminary: Multi-Head Softmax Attention)
- Inference: Inputs/outputs treated as token sequences with 1D (t) and capped dynamics based on "3.5T high-quality tokens" and "context sequence length is set to 4096." (Section 3.1); Static attention and Direct state inferred from the standard softmax attention over sequence inputs (Section 2.1).

### Task: English benchmark evaluation (Hellaswag)
- "including Hellaswag (Zellers et al., 2019) for English" (Section 3.1, Evaluation)
- Inference: Inputs/outputs treated as token sequences with 1D (t) and capped dynamics based on "3.5T high-quality tokens" and "context sequence length is set to 4096." (Section 3.1); Static attention and Direct state inferred from the standard softmax attention over sequence inputs (Section 2.1).

### Task: General knowledge benchmark evaluation (MMLU)
- "MMLU (Hendrycks et al., 2020) for general knowledge" (Section 3.1, Evaluation)
- Inference: Inputs/outputs treated as token sequences with 1D (t) and capped dynamics based on "3.5T high-quality tokens" and "context sequence length is set to 4096." (Section 3.1); Static attention and Direct state inferred from the standard softmax attention over sequence inputs (Section 2.1).

### Task: Math reasoning benchmark evaluation (GSM8k)
- "GSM8k (Cobbe et al., 2021) for math reasoning" (Section 3.1, Evaluation)
- Inference: Inputs/outputs treated as token sequences with 1D (t) and capped dynamics based on "3.5T high-quality tokens" and "context sequence length is set to 4096." (Section 3.1); Static attention and Direct state inferred from the standard softmax attention over sequence inputs (Section 2.1).

### Task: Coding benchmark evaluation (HumanEval)
- "HumanEval (Chen et al., 2021) for coding" (Section 3.1, Evaluation)
- Inference: Inputs/outputs treated as token sequences with 1D (t) and capped dynamics based on "3.5T high-quality tokens" and "context sequence length is set to 4096." (Section 3.1); Static attention and Direct state inferred from the standard softmax attention over sequence inputs (Section 2.1).

### Task: Chinese proficiency benchmark evaluation (C-eval, CMMLU)
- "C-eval (Huang et al., 2024) and CMMLU (Li et al., 2023) for Chinese proficiency" (Section 3.1, Evaluation)
- Inference: Inputs/outputs treated as token sequences with 1D (t) and capped dynamics based on "3.5T high-quality tokens" and "context sequence length is set to 4096." (Section 3.1); Static attention and Direct state inferred from the standard softmax attention over sequence inputs (Section 2.1).

### Task: Long-context benchmark evaluation (RULER)
- "We evaluate models on the RULER benchmark (Hsieh et al., 2024)" (Section 4.4, SDPA Output Gating Facilitates Context Length Extension)
- "extend the context length to 128k." (Section 4.4, SDPA Output Gating Facilitates Context Length Extension)
- Inference: Inputs/outputs treated as token sequences with 1D (t) and capped dynamics based on "3.5T high-quality tokens" and the stated context-length limits (Sections 3.1 and 4.4); Static attention and Direct state inferred from the standard softmax attention over sequence inputs (Section 2.1).
