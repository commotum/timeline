# Positional Encoding via Token-Aware Phase Attention (Year not specified in the paper)
Source: TAPA- Positional Encoding via Token-Aware Phase Attention.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (long-context language modeling) | token sequences (text documents) | 1D (t) | Capped | Static (inferred) | Direct (inferred) | tokens (next-token predictions) (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers one task: long-context language modeling over text token sequences. The input domain is 1D (t) with capped length, explicitly evaluated from 1k to 64k context windows. Based on the described causal-mask transformer setup, attention/state are best classified as Static and Direct (inferred). The output is inferred as token predictions in 1D (t), with capped output length tied to context windows.

## Evidence
### Task: generation (long-context language modeling)
- "RoPE's distance bias is harmful for long-context language modeling, as it hurts model's ability in feeling long-range dependencies and leveraging distant information." (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA))
- "To measure models' performance at different context lengths, we consider segmentation of each document with context window size varying from 1k to 64k in the dyadic fashion." (Section 4.3 Long-Context Evaluation)
- "**Table 1** Test perplexities on PG19 test set of LLaMA3 7B transformers first pretrained on 8k context length and further fine-tuned on 32k, and evaluated on  $1k\sim64k$ ." (Table 1)
- "we preserve the causal masking mechanism and focus on improving the positional encoding itself." (Section 5.3 Non-RoPE Approaches to Positional Extrapolation)
- Inference: Attention Dynamic is Static (inferred) because the paper keeps causal masking and does not describe runtime retrieval or observation selection. State Dynamic is Direct (inferred) because it keeps the standard transformer formulation ("we make no changes to transformer architecture other than removing RoPE and replacing transformer's inner product attention with Equation (12)", Section 4.1 Pretraining). Output, Out Dimension, and Out Dynamics are inferred as token predictions in 1D (t) with capped length because the task is language modeling evaluated via token-window perplexity up to 64k.
