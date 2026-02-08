# Shortformer: Better Language Modeling Using Shorter Inputs (Not specified in the paper)
Source: Shortformer- Better Language Modeling using Shorter Inputs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Evaluation (perplexity scoring) | Token sequence (given text) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Perplexity score | 0D (inferred) | Fixed (inferred) |
| Generation (next-token language modeling) | Token sequence context | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Generated token sequence | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper explicitly defines two inference tasks for language models: evaluation and generation, both over text token sequences. The supported input modality is linear token data, which maps to 1D (t), while outputs span both scalar sequence-level scoring (perplexity) and token-sequence generation. The model interface is capped by subsequence length `L`, and the paper states it does not study open-ended generation, supporting capped dynamics. Attention is static and state is constructed (inferred) based on fixed-window processing plus explicit cache reuse of prior representations in PIA models.

## Evidence
### Task: Evaluation (perplexity scoring)
- "During inference, language models can be used for two distinct tasks: generation and evaluation." (Section 2, Background and Experimental Setup)
- "In evaluation, a model assigns a perplexity score to a given sequence." (Section 2, Evaluation vs. Generation)
- "Transformer language models map a list of tokens  $x_{n-L:n-1}$  to a probability distribution over the next token  $x_n$ . We refer to the list of tokens as the *current input subsequence* (whose length is L)." (Section 2, Background and Experimental Setup)
- "memory constraints limit a language model to handling at most a few thousand tokens at once" (Section 1, Introduction)
- "Therefore, all our PIA models use a cache, where representations from the previous forward pass are stored and attended to in the next forward pass." (Section 5.2, PIA Enables Caching)
- Inference: `1D (t)` is inferred from ordered token subsequences; `Capped` is inferred from the explicit maximum subsequence length `L`; `Static` attention is inferred because the considered context is fixed by predefined subsequences/stride rather than runtime retrieval; `Constructed` state is inferred from explicit caching of prior representations; `0D` and `Fixed` output are inferred because evaluation is defined as assigning a single perplexity score to a sequence.

### Task: Generation (next-token language modeling)
- "During inference, language models can be used for two distinct tasks: generation and evaluation." (Section 2, Background and Experimental Setup)
- "In generation, a model generates a new sequence, as in demonstrations of GPT-3 (Brown et al., 2020)." (Section 2, Evaluation vs. Generation)
- "Generation is done only with a sliding window with stride S = 1, which we refer to as token-by-token generation." (Section 2, Evaluation vs. Generation)
- "In this paper we do not consider open-ended generation; we generate the dev. set, and for next-token prediction we use the ground truth token." (Section 2, footnote)
- "Therefore, all our PIA models use a cache, where representations from the previous forward pass are stored and attended to in the next forward pass." (Section 5.2, PIA Enables Caching)
- Inference: `1D (t)` is inferred from sequential token inputs/outputs; `Capped` input/output dynamics are inferred from bounded subsequence length `L` and the paper’s explicit non-open-ended generation setup; `Static` attention is inferred from fixed sliding-window/context definitions; `Constructed` state is inferred from explicit cache-based reuse of prior internal representations.
