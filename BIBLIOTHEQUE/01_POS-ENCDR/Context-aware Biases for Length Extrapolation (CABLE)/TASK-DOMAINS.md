# Context-aware Biases for Length Extrapolation (Not specified in the paper)
Source: Context-aware Biases for Length Extrapolation (CABLE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction | tokens (text sequence) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (next-token predictions) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Masked language modeling (MLM) | tokens with masks (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens for masked positions (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Long-context retrieval | text queries and documents (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | ranked documents / relevance scores (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates CABLE on text-only NLP tasks: GPT-2 next-token prediction, BERT masked language modeling, and long-context retrieval on MLDR. It reports fixed training lengths (1024 for GPT-2; 512 for BERT), so the task domains are 1D (t) with capped dynamics, and attention/state attributes are inferred from the token-sequence attention formulation. Retrieval is evaluated with nDCG@10, so its output is treated as ranked documents or relevance scores (inferred).

## Evidence
### Task: Next-token prediction
- "using GPT-2 variants for next-token prediction" (Contributions)
- "we train the models with sequence length of 1024." (Section 4.2 Settings)
- "tokens in the input sequence" (Section 3 Proposed Method)
- Inference: Treated inputs/outputs as token sequences with 1D (t) and capped dynamics, and attention/state as static/direct, based on "tokens in the input sequence" and the fixed sequence length. (Sections 3 and 4.2)

### Task: Masked language modeling (MLM)
- "train the models using only the masked language modeling (MLM) objective" (Section 5.5 Bidirectional Models)
- "maximum sequence length of 512" (Section 5.5 Bidirectional Models)
- "tokens in the input sequence" (Section 3 Proposed Method)
- Inference: Treated inputs/outputs as token sequences with masks, 1D (t), capped dynamics, and static/direct attention/state based on the MLM setup and token-sequence attention formulation. (Sections 3 and 5.5)

### Task: Long-context retrieval
- "a retrieval benchmark consisting of over 200,000 long documents." (Section 5.5 Bidirectional Models)
- "evaluate the fine-tuned models on the MLDR test set using nDCG@10" (Section 5.5 Bidirectional Models)
- "trained at sequence length 512 and evaluated on longer inputs." (Table 3 caption)
- Inference: Treated inputs as text queries and documents and outputs as ranked documents/relevance scores, with 1D (t) capped dynamics and static/direct attention/state inferred from the retrieval setup and sequence-length constraints. (Section 5.5; Table 3 caption)
