# CRoPE: Efficient Parametrization of Rotary Positional Embedding (Year not specified)
Source: CRoPE- Efficient Parametrization of Rotary Positional Embedding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| token comparison | tokens / token embeddings | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Not specified in the paper. | attention scores/weights (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| position comparison | position encodings | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Not specified in the paper. | attention scores/weights (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| token-dependent position comparison | tokens with positions | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Not specified in the paper. | attention weights over positions | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper explicitly analyzes three illustrative attention tasks over token sequences: token comparison, position comparison, and token-dependent position comparison. These tasks operate over 1D token positions (inferred) with Dynamic attention (inferred) based on token/position values, while input/output dynamics and state construction are not specified. Downstream application tasks are not explicitly stated beyond these attention mechanisms.

## Evidence
### Task: token comparison
- "One basic mechanisms is token comparison, e.g. attending to similar tokens." (Section "Simple token comparison")
- "which takes larger value when the tokens x_m and x_n are similar." (Section "Simple token comparison")
- Inference: Classified inputs/outputs as 1D (t), outputs as attention scores/weights, and attention as Dynamic because the task compares indexed tokens x_m, x_n and attends based on token similarity.

### Task: position comparison
- "Another fundamental mechanisms is position comparison, e.g. attending to near positions." (Section 4.2 "Simple position comparison")
- "which takes larger value when the position encodings p_m and p_n are similar." (Section 4.2 "Simple position comparison")
- Inference: Classified inputs/outputs as 1D (t), outputs as attention scores/weights, and attention as Dynamic because it compares indexed positions m, n via position encodings and attends to near positions.

### Task: token-dependent position comparison
- "The most basic task is to have a token-dependent position comparison." (Section 4.3 "Token-dependent position comparison")
- "At the i-th token, the ideal attention weight depends on the token value." (Section 4.3 "Token-dependent position comparison")
- "desired attention weights focus on the i + 1-th token" (Figure 3 caption)
- Inference: Classified inputs/outputs as 1D (t) and attention as Dynamic because attention weights depend on token value and focus on positions i+1 or i+2.
