# Scaling Laws for Neural Language Models (Year not specified in the paper)
Source: Scaling Laws for Neural Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoregressive language modeling (next-token prediction) | Text tokens in context | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token probabilities/tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper studies a single core task: autoregressive language modeling on text. Inputs are tokenized text contexts and the optimization target is autoregressive cross-entropy over tokens, with context length typically set to 1024 and also tested at shorter contexts. From this, the task is best characterized as 1D (t) with capped interface dynamics due to explicit context-window limits. Attention and state dynamics are inferred as Static and Direct because the model processes a predefined token window and is trained as a reactive next-token predictor.

## Evidence
### Task: Autoregressive language modeling (next-token prediction)
- "The test loss of a Transformer trained to autoregressively model language can be predicted using a power-law when performance is limited by only either the number of non-embedding parameters N, the dataset size D, or the optimally allocated compute budget  $C_{\min}$  (see Figure 1):" (Section 1.2 Summary of Scaling Laws)
- "We optimize the autoregressive log-likelihood (i.e. cross-entropy loss) averaged over a 1024-token context, which is also our principal performance metric." (Section 2 Background and Methods)
- "We include  $n_{\rm ctx}$  tokens in the input context, with  $n_{\rm ctx}=1024$  except where otherwise noted." (Section 2.1 Parameter and Compute Scaling of Transformers)
- Inference: `In Dimension`, `Out Dimension`, `In Dynamics`, `Out Dynamics`, `Attention Dynamic`, `State Dynamic`, and token-level `Output` are inferred from the autoregressive token objective and explicit context-window constraint (`n_ctx=1024` with shorter-context experiments). This supports 1D (t) token sequences with capped context, Static attention policy in the glossary sense (fixed window processed at runtime), and Direct state for next-token prediction.
