# MemoryLLM: Plug-n-Play Interpretable Feed-Forward Memory for Transformers (2026)
Source: MemoryLLM- Plug-n-Play Interpretable Feed-Forward Memory for Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Perplexity evaluation (C4) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Recall/retrieval of known information (Wikitext-2) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Recall/retrieval of known information (LAMBDA) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Recall/retrieval of known information (SiQA) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Recall/retrieval of known information (ARC-Easy) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Logical/causal/inferential thinking (HellaSwag) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Logical/causal/inferential thinking (Winogrande) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Logical/causal/inferential thinking (BoolQ) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Logical/causal/inferential thinking (PIQA) | tokens (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | tokens (text) (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates MemoryLLM on text-domain benchmarks, including a perplexity evaluation on C4 and two task families: recall/retrieval and logical/causal/inferential thinking tasks. The model operates over token sequences, so inputs and outputs are 1D (t) with capped input length (sequence length 2048), and the architecture implies static attention over a fixed context and constructed state via token-wise retrieval memory (inferred). Task-specific input/output formats beyond token sequences are not explicitly specified for the listed benchmarks.

## Evidence
### Task: Perplexity evaluation (C4)
- "Table 8 Performance comparison of MemoryLLM-1B with uniform low-rank SVD compression of ToLs across 24 layers." (Section D.2)
- "C4-PPL" (Table 8, Section D.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Recall/retrieval of known information (Wikitext-2)
- "tasks which heavily rely on recall or retrieval of known information" (Section 3.2)
- "(e.g., wikitext-2, LAMBDA, SiQA, ARC-Easy)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Recall/retrieval of known information (LAMBDA)
- "tasks which heavily rely on recall or retrieval of known information" (Section 3.2)
- "(e.g., wikitext-2, LAMBDA, SiQA, ARC-Easy)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Recall/retrieval of known information (SiQA)
- "tasks which heavily rely on recall or retrieval of known information" (Section 3.2)
- "(e.g., wikitext-2, LAMBDA, SiQA, ARC-Easy)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Recall/retrieval of known information (ARC-Easy)
- "tasks which heavily rely on recall or retrieval of known information" (Section 3.2)
- "(e.g., wikitext-2, LAMBDA, SiQA, ARC-Easy)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Logical/causal/inferential thinking (HellaSwag)
- "tasks that require logical, causal, or inferential thinking" (Section 3.2)
- "(e.g., HellaSwag, Winogrande, BoolQ, PIQA)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Logical/causal/inferential thinking (Winogrande)
- "tasks that require logical, causal, or inferential thinking" (Section 3.2)
- "(e.g., HellaSwag, Winogrande, BoolQ, PIQA)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Logical/causal/inferential thinking (BoolQ)
- "tasks that require logical, causal, or inferential thinking" (Section 3.2)
- "(e.g., HellaSwag, Winogrande, BoolQ, PIQA)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)

### Task: Logical/causal/inferential thinking (PIQA)
- "tasks that require logical, causal, or inferential thinking" (Section 3.2)
- "(e.g., HellaSwag, Winogrande, BoolQ, PIQA)" (Section 3.2)
- Inference: Inputs/outputs are token sequences with 1D (t), input dynamics are Capped, attention is Static, and state is Constructed based on "An input text T = {t1, t2, ..., tM} of M tokens is transformed to embedding vectors", "projection back to tokens", "Sequence Length 2048", "self-attention module takes the snapshot X_L of residual stream and transforms it with contextual information", and "interpreting the up-projection (key) and down-projection (value) matrices as neural retrieval memory." (Sections 1, 2.1, 2.2.1, 2.3.1; Table 3)
