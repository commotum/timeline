# Adaptive Attention Span in Transformers (Not specified in the paper)
Source: Adaptive Attention Span in Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Character-level language modeling | tokens (characters) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | token probabilities | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper studies character-level language modeling, defined as assigning probabilities to sequences of tokens (characters). The task operates over 1D token sequences with capped context sizes (e.g., a maximum span up to 8k characters), so both input and output dynamics are capped. The attention mechanism uses learned per-head spans (treated here as static), though the paper also describes a dynamic-span extension; the model operates directly over the token sequence without an explicit constructed external state.

## Evidence
### Task: Character-level language modeling
- "We show the effectiveness of our approach on the task of character level language modeling" (Abstract)
- "Language modeling is the problem of assigning a probability to a sequence of tokens" (Section 2.1)
- "using a maximum context of 8k characters" (Abstract)
- "At train time, we use a block of 512 consecutive characters" (Section 3 Experiments)
- "For each head, we add a masking function to control for the span of the attention." (Section 2.2)
- "the span parameter z_t of an attention head is then a function of the input" (Section 2.2, Dynamic attention span)
- Inference: Interpreted token/character sequences as 1D (t) input/output and capped dynamics because the model uses a maximum context span and fixed-length blocks; attention is marked Static because the base adaptive span uses a learned per-head mask parameter z (the paper separately describes a dynamic-span extension where z_t depends on input), and state is Direct because predictions are computed from the token sequence within the context span rather than an explicit external state. Supporting text includes the token-sequence definition, the maximum context span, fixed-length training blocks, the masking function description, and the dynamic-span statement (Abstract; Sections 2.1, 2.2, 3 Experiments).
