# Mano: Restriking Manifold Optimization for LLM Training (Not specified in the paper)
Source: Mano- Restriking Manifold Optimization for LLM Training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (LLM pretraining) (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | next-token predictions / tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates LLM pretraining on text corpora such as C4 and Pile and reports test perplexity, which supports a language modeling/next-token prediction task. The use of tokenizers and a stated sequence length of 1024 suggest 1D token sequences with capped length. The paper does not specify attention or state dynamics for the task.

## Evidence
### Task: Language modeling (LLM pretraining) (inferred)
- "In this paper, we studied the pretraining performance of five popular models" (Section 5.1 Experiment Setup)
- "two common text corpus, including C4 and Pile" (Section 5.1 Experiment Setup)
- "We present the pretraining dynamics of LLMs in test perplexity" (Section 5.2 Experiment Results)
- "The LLaMA models are tokenized using the T5 tokenizer" (Appendix B.1 Hyperparameters)
- "| Seq-len           | 1024                 | 1024                 | 1024                 | 1024                 | 1024                 |" (Appendix B.1, Table 5)
- Inference: The task is language modeling/next-token prediction over token sequences, with 1D (t) inputs and outputs capped by the 1024 sequence length; this is inferred from pretraining on text corpora, tokenizer usage, test perplexity reporting, and the stated sequence length.
