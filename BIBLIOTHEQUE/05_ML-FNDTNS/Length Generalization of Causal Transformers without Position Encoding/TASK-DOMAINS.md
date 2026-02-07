# Length Generalization of Causal Transformers without Position Encoding (Not specified in the paper)
Source: Length Generalization of Causal Transformers without Position Encoding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Long sequence language modeling | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Passkey retrieval (synthetic) | long sequence of tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | 5-digit passkey number | 1D (t) (inferred) | Fixed (inferred) |
| Long-context understanding (LongBench) | long-context text (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text responses (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Commonsense reasoning evaluation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates long sequence language modeling, a synthetic passkey retrieval task, and LongBench long-context understanding, plus a separate commonsense reasoning evaluation. Explicitly described inputs/outputs are token sequences and 5-digit passkeys, while the 1D (t) structure and capped context dynamics for LM-style tasks are inferred from the evaluation setup. Attention and state dynamics are not specified and are inferred as Static/Direct for the causal LM decoding setting.

## Evidence
### Task: Long sequence language modeling
- "We conduct length generalization experiments on long sequence language modeling, synthetic tasks (passkey retrieval), and real-world long context tasks (LongBench)." (Section 4)
- "we test our NoPE-based methods and RoPE-based baselines on PG19 (Rae et al., 2020) and proof-pile (Azerbayev et al., 2022) datasets." (Section 4.2)
- "evaluate on 2M tokens using sliding window evaluation (S=256)" (Section 4.2)
- Inference: Inferred token-sequence input/output, 1D (t) structure, capped context dynamics, and static/direct processing from the language modeling setup and token-based sliding window evaluation. (Section 4.2)

### Task: Passkey retrieval (synthetic)
- "A synthetic task is constructed in Landmark Attention (Mohtashami and Jaggi, 2023b) called \"Passkey Retrieval\"." (Section 4.3)
- "The task is to retrieve a randomly placed passkey from a long sequence of tokens" (Section 4.3)
- "the passkey is a randomly sampled number of 5 digits" (Section 4.3)
- Inference: Inferred 1D (t) sequence structure, capped context dynamics, static/direct processing, and fixed-length output from the token-sequence setup and 5-digit passkey definition. (Section 4.3)

### Task: Long-context understanding (LongBench)
- "LongBench (Bai et al., 2023) is a comprehensive assessment of the long context understanding capabilities of large language models." (Section 4.4)
- "We test all models using beam search decoding with beam size 5." (Section 4.4)
- "The evaluation context size is set to the model context window accordingly" (Section 4.4)
- Inference: Inferred long-context text input/output in 1D (t) with capped context, and static/direct processing from the long-context LM decoding setup. (Section 4.4)

### Task: Commonsense reasoning evaluation
- "Following TinyLlama, we evaluate the commonsense reasoning ability of the NoPE model and report acc_norm in Table 1." (Section 4.1)
- "Table 1: Commonsense reasoning ability of the pre-trained base models." (Section 4.1)
