# Base of RoPE Bounds Context Length (Not specified in the paper.)
Source: Base of RoPE Bounds Context Length.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language modeling | token sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | next-token predictions (inferred) | 1D (t) (inferred) | Capped (inferred) |
| retrieval (long-context question answering) | context sentences + questions | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | answers | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates large language models on language modeling (perplexity) and long-context retrieval question answering (Long-eval and needle-in-a-haystack), all within text sequences. The inputs and outputs are 1D (t) sequences with capped context lengths inferred from the paper's focus on sequential data and fixed context windows. Attention is inferred to be static over the provided context, while state dynamics are not specified.

## Evidence
### Task: language modeling
- "For attention mechanism in language modeling" (Section 4 Theory Perspective)
- "Perplexity: we use PG19 dataset (Rae et al., 2019)" (Section 5.1 Experiments Setup)
- Inference: Treated inputs/outputs as 1D token sequences with capped length and static attention because the paper emphasizes "processing sequential data" (Section 1 Introduction), uses fixed "context length of 32k" (Section 3 Motivation), and defines standard attention computation ("The core component of it is the calculation of the attention mechanism.", Section 2.1 Attention and RoPE).

### Task: retrieval (long-context question answering)
- "Retrieval: in addition to perplexity, we also adopt retrieval since it represents the real long-context understanding ability of LLMs." (Section 5.1 Experiments Setup)
- "asks the model to answer questions based on a specific sentence within the context" (Section 5.1 Experiments Setup)
- "NIH requires the model to retrieve information from various positions in the long context." (Section 5.1 Experiments Setup)
- Inference: Treated inputs as 1D sequences with capped context and static attention because the paper describes "processing sequential data" (Section 1 Introduction) and fixed "context length of 32k" (Section 3 Motivation), alongside standard attention computation ("The core component of it is the calculation of the attention mechanism.", Section 2.1 Attention and RoPE).
