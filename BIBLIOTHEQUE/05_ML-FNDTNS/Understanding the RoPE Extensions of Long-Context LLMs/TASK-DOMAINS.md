# Understanding the RoPE Extensions of Long-Context LLMs: An Attention Perspective (Year not specified in the paper)
Source: Understanding the RoPE Extensions of Long-Context LLMs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prediction (long-context language modeling via perplexity evaluation) | Long text token sequences from Proofpile | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Perplexity score | 0D (inferred) | Fixed (inferred) |
| Retrieval (Needle-in-a-Haystack sentence recall) | Long document token sequences with an embedded needle sentence at an arbitrary location | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Recalled needle sentence (answer tokens) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates two long-context, text-only tasks: perplexity-based language modeling and Needle-in-a-Haystack sentence recall. Both tasks operate over token order in 1D (t) and are tested across bounded context lengths, so input dynamics are classified as Capped (inferred). From the decoder-only LLM setup and attention analyses over provided context, attention is classified as Static and state as Direct (both inferred). The outputs include a scalar perplexity score (0D) and recalled text tokens (1D), with output dynamics Fixed for perplexity and Capped for needle recall (inferred).

## Evidence
### Task: Prediction (long-context language modeling via perplexity evaluation)
- "Following existing works (Chen et al., 2023a; Peng et al., 2023; Fu et al., 2024), we use the perplexity test (dubbed PPL) as the primary evaluation and the Needle-in-a-Haystack test as a more challenging evaluation." (Section 2.3 Long-Context Evaluations)
- "The perplexity is a primary measure that reflects a model's ability to handle long texts." (Section 2.3 Long-Context Evaluations)
- "We obtain the perplexity on the Proofpile (Azerbayev et al., 2022) dataset." (Section 2.3 Long-Context Evaluations)
- "NTK can extrapolate from 4K to 128K, whereas PI and YaRN can extrapolate to 62K." (Section 3 RoPE Extensions on PPL)
- Inference: The task input is long token-ordered text and tested with bounded maximum lengths (e.g., "4K to 128K"), so In Dimension is 1D (t) and In Dynamics is Capped (inferred). Because this is decoder-only language modeling with self-attention over the provided context, Attention Dynamic is Static and State Dynamic is Direct (inferred). Since the reported task outcome is perplexity, Out Dimension is 0D and Out Dynamics is Fixed (inferred).

### Task: Retrieval (Needle-in-a-Haystack sentence recall)
- "The Needle-in-a-Haystack test (dubbed Needle) (Kamradt, 2023) requires LLMs to accurately recall a specific sentence (the Needle) embedded at an arbitrary location within a long document (the haystack)." (Section 2.3 Long-Context Evaluations)
- "As shown in Figure 4(a-d), LLaMa-2-7B with RoPE extensions can pass more needle tests than the RoPE. However, as the context length increases, some tests fail, resulting in needle retrieval errors." (Section 4 RoPE Extensions on Needle)
- "The x-axis represents the length of the document, while y-axis indicates the depth percentage, showing the needle's position within the document." (Figure 4 caption / Section 4)
- "A red cell indicates that the model fails to recall the information in the needle, whereas a green cell indicates success." (Figure 4 caption / Section 4)
- Inference: The haystack/needle setup is long token-sequence text indexed by position in the document, so In Dimension is 1D (t) (inferred). The evaluation varies document length but remains within tested maximum lengths, so In Dynamics is Capped (inferred). Given decoder-only self-attention over supplied context, Attention Dynamic is Static and State Dynamic is Direct (inferred). The recalled sentence is text tokens, so Out Dimension is 1D (t) and Out Dynamics is Capped (both inferred).
