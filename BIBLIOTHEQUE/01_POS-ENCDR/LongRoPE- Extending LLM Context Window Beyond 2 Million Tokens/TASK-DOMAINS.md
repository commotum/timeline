# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (Not specified in the paper.)
Source: LongRoPE- Extending LLM Context Window Beyond 2 Million Tokens.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (language modeling / perplexity) | long documents | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (next-token predictions) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| retrieval (passkey retrieval) | long document with embedded passkey | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | passkey (five-digit number) | 0D (inferred) | Fixed (inferred) |
| question answering (standard LLM benchmarks) (inferred) | standard LLM benchmarks (ARC-Challenge, HellaSwag, MMLU, TruthfulQA) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | benchmark answers / choices (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates tasks spanning long-sequence language modeling (perplexity), passkey retrieval, and standard LLM benchmark QA within a 4096-token window. Inputs are 1D token sequences with capped context lengths up to 2048k tokens, while outputs are predicted tokens or single-answer values (0D/1D) with fixed or capped dynamics (inferred). Because LongRoPE retains the original LLM architecture, attention is treated as static and state as direct rather than constructed (inferred).

## Evidence
### Task: prediction (language modeling / perplexity)
- "perplexity of extended-context LLMs on long documents" (Section 4.1. Setup)
- "minimum next token prediction loss, L (i.e., the perplexity)" (Section 3.1. Problem Formulation)
- "extends the context window of pre-trained LLMs to an impressive 2048k tokens" (Abstract)
- Inference: Inputs/outputs treated as 1D token sequences with capped length; attention/state inferred from "retain the original architecture with minor modifications to the positional embedding" (Abstract).

### Task: retrieval (passkey retrieval)
- "Passkey retrieval task that measures a model's ability to retrieve a simple passkey" (Section 4.1. Setup)
- "retrieve a random passkey (i.e., a five-digit number) hidden in long document." (Section 4.2. Main Results)
- "random location uniformly distributed across the evaluation context length." (Section 4.2. Main Results)
- "The pass key is 17865. Remember it. 17865 is the pass key." (Appendix A.1. Settings)
- Inference: Input treated as a capped 1D sequence and output as a fixed single value because the passkey is a "five-digit number" and evaluation uses a fixed context window; attention/state inferred from "retain the original architecture with minor modifications to the positional embedding" (Abstract; Section 4.2. Main Results).

### Task: question answering (standard LLM benchmarks) (inferred)
- "Standard LLM benchmarks within a short 4096 context window size." (Section 4.1. Setup)
- "We use 25-shot ARC-Challenge, 10-shot HellaSwag, 5-shot MMLU, and 0-shot TruthfulQA." (Section 4.2. Main Results)
- Inference: Treated these benchmarks as question-answering with single-answer outputs and capped 1D inputs based on the benchmark evaluation setting and "short 4096 context window size"; attention/state inferred from "retain the original architecture with minor modifications to the positional embedding" (Abstract).
