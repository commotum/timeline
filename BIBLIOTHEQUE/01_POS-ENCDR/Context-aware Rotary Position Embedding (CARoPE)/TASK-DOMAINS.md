# Context-aware Rotary Position Embedding (CARoPE) (Not specified in the paper)
Source: Context-aware Rotary Position Embedding (CARoPE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction | tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | next token (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates a single language modeling task: next-token prediction on text from FineWeb-Edu. Inputs are token sequences and the task operates over ordered sequences, with a capped context length (512 in training and 512/1024 in evaluation) inferred from the reported sequence lengths. The task uses standard transformer self-attention without any runtime selection mechanism described (Static, inferred) and maps inputs directly to next-token outputs (Direct, inferred).

## Evidence
### Task: Next-token prediction
- "We evaluate CARoPE on the FineWeb-Edu-10B dataset using GPT-2 variants trained on next-token prediction tasks." (Abstract)
- "x_t in R^d is the embedding of the token at position t." (2 Proposed Method)
- "where m is the sequence position" (2 Proposed Method)
- "512<br>1024" (Table 1)
- "with a context length of 512." (Table 1 caption)
- Inference: Input/Output dimensions labeled 1D (t) and dynamics labeled Capped are inferred from the sequence position and the reported sequence/context lengths (2 Proposed Method; Table 1; Table 1 caption). Output as a next token, and Attention Dynamic = Static and State Dynamic = Direct are inferred from the next-token prediction setup and standard transformer self-attention without any runtime selection or external memory described (Abstract; 2 Proposed Method).
