# BEHIND ROPE: How Does Causal Mask Encode Positional Information? (Not specified in the paper.)
Source: Behind RoPE- How Does Causal Mask Encode Positional Information-.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language modeling (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper analyzes Transformer decoders and LLMs in a language-modeling setting using token sequences and causal masking. The task domain is 1D (t) token sequences with a fixed context window (1024), implying capped dynamics and static attention from the causal mask. Outputs are token predictions from the same 1D domain and are inferred from the language-model training setup. No additional task domains beyond language modeling are specified.

## Evidence
### Task: language modeling (inferred)
- "Let the input token embeddings be  $X^{(0)} = [x_1^{(0)}, \cdots, x_n^{(0)}] \in \mathbb{R}^{n \times d}$ , where n is the number of input tokens" (Section 3.1 Preliminaries)
- "operation  $Causal(\cdot)$  applies a strictly upper-triangular mask to prevent attention to future positions." (Section 3.1 Preliminaries)
- "we trained a model based on the Llama-3 architecture (Grattafiori et al., 2024) having 1.5B parameters (22 layers, hidden dimension 2048, head dimension 64) on 20 billion tokens from the Fineweb-Edu corpus (Penedo et al., 2024)." (Section 4.2 Analysis of a Trained Model Without Positional Encoding)
- "and a context length of 1024." (Section 4.2 Analysis of a Trained Model Without Positional Encoding)
- "Since input embeddings in the language model contain no positional information" (Section 4.2 Analysis of a Trained Model Without Positional Encoding)
- Inference: Labeled the task as language modeling and set outputs to tokens with 1D (t), capped dynamics, static attention, and direct state because the paper describes a Transformer decoder language model trained on token sequences with a fixed context length and a causal mask (see the quoted statements about input tokens, language model, context length, and causal mask).
