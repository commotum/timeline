# YaRN: Efficient Context Window Extension of Large Language Models (Not specified in the paper.)
Source: YaRN- Efficient Context Window Extension of Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Long sequence language modeling | Long text sequences (GovReport and Proof-pile samples with up to 128k tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Perplexity scores | 0D (inferred) | Fixed (inferred) |
| Passkey retrieval | Text containing a five-digit passkey among otherwise meaningless text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Five-digit passkey | 1D (t) (inferred) | Fixed (inferred) |
| ARC-Challenge (25-shot benchmark) | Text benchmark prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Benchmark answer/score for ARC-c (inferred) | 0D (inferred) | Fixed (inferred) |
| HellaSwag (10-shot benchmark) | Text benchmark prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Benchmark answer/score for HellaSwag (inferred) | 0D (inferred) | Fixed (inferred) |
| MMLU (5-shot benchmark) | Text benchmark prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Benchmark answer/score for MMLU (inferred) | 0D (inferred) | Fixed (inferred) |
| TruthfulQA (0-shot benchmark) | Text benchmark prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Benchmark answer/score for TruthfulQA (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates text-only tasks in three evaluation groups: long-sequence language modeling, passkey retrieval, and a four-task standardized benchmark suite (ARC-Challenge, HellaSwag, MMLU, TruthfulQA). Across these, the justified input structure is token sequences, mapped to 1D (t) with capped context windows up to explicit maxima (e.g., 128k). The attention/state assignments are inferred as Static and Direct from the described autoregressive transformer setup and context-windowed evaluation protocol.

## Evidence
### Task: Long sequence language modeling
- "The evaluations focus on three aspects:

- 1. the perplexity scores of fine-tuned models with extended context window," (Section 4.3 Evaluation)
- "To evaluate the long sequence language modeling performances, we use the GovReport [18] and Proof-pile [4] datasets both of which contain many long sequence samples." (Section 4.3.1 Long Sequence Language Modeling)
- "We selected 10 random samples from Proof-pile with at least 128k tokens each and evaluated the perplexity of each of these samples when truncated at 2k steps from a sequence length of 2k tokens through 128k tokens." (Section 4.3.1 Long Sequence Language Modeling)
- Inference: 1D (t), Capped, Static, and Direct are inferred from token-sequence processing with explicit context-window bounds and autoregressive forward passes ("multiple forward-passes are performed with varying sequence lengths from 1 to the maximal context size"; "autoregressive generation" in Section 3.3). 0D output and Fixed output dynamics are inferred because this row reports perplexity scores as scalar evaluation outputs.

### Task: Passkey retrieval
- "The passkey retrieval task as defined in [25] measures a model's ability to retrieve a simple passkey (i.e., a five-digit number) from amongst a large amount of otherwise meaningless text." (Section 4.3.2 Passkey Retrieval)
- "For our evaluation of the models, we performed 10 iterations of the passkey retrieval task with the passkey placed at a random location uniformly distributed across the evaluation context window on different context window sizes ranging from 8k to 128k." (Section 4.3.2 Passkey Retrieval)
- Inference: 1D (t), Capped, Static, and Direct are inferred from token-sequence context-window evaluation and autoregressive operation (Sections 3.3 and 4.3.2). 1D (t) output with Fixed dynamics is inferred from the explicit "five-digit number" output requirement.

### Task: ARC-Challenge (25-shot benchmark)
- "The Hugging Face Open LLM Leaderboard [19] compares a multitude of LLMs across a standardized set of four public benchmarks." (Section 4.3.3 Standardized Benchmarks)
- "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)
- Inference: Text prompts, 1D (t), Capped, Static, and Direct are inferred from the paper's framing of transformer LLMs for NLP tasks (Section 1) and autoregressive context-windowed inference (Section 3.3). 0D Fixed output is inferred because this benchmark is reported as a scalar ARC-c result in Table 3.

### Task: HellaSwag (10-shot benchmark)
- "The Hugging Face Open LLM Leaderboard [19] compares a multitude of LLMs across a standardized set of four public benchmarks." (Section 4.3.3 Standardized Benchmarks)
- "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)
- Inference: Text prompts, 1D (t), Capped, Static, and Direct are inferred from the same LLM/autoregressive context-window setup described in Sections 1 and 3.3. 0D Fixed output is inferred because HellaSwag is reported as a scalar benchmark score in Table 3.

### Task: MMLU (5-shot benchmark)
- "The Hugging Face Open LLM Leaderboard [19] compares a multitude of LLMs across a standardized set of four public benchmarks." (Section 4.3.3 Standardized Benchmarks)
- "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)
- Inference: Text prompts, 1D (t), Capped, Static, and Direct are inferred from Sections 1 and 3.3 (LLM NLP framing and autoregressive sequence processing under a maximum context size). 0D Fixed output is inferred because MMLU is reported as a scalar benchmark score in Table 3.

### Task: TruthfulQA (0-shot benchmark)
- "The Hugging Face Open LLM Leaderboard [19] compares a multitude of LLMs across a standardized set of four public benchmarks." (Section 4.3.3 Standardized Benchmarks)
- "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)
- Inference: Text prompts, 1D (t), Capped, Static, and Direct are inferred from the model/task framing and autoregressive context-window behavior in Sections 1 and 3.3. 0D Fixed output is inferred because TruthfulQA is reported as a scalar benchmark score in Table 3.
