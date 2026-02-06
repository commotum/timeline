# A Length-Extrapolatable Transformer (Not specified in the paper.)
Source: A Length-Extrapolatable Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (causal language modeling) | text tokens | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | tokens (next-token prediction) (inferred) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper evaluates a single text task: causal language modeling on token sequences, reporting perplexity across varying sequence lengths. The inputs and outputs are 1D token sequences with fixed causal or blockwise masks (static attention), and no explicit constructed state is described. The authors emphasize length extrapolation and claim the model can handle any input length, so the sequence dynamics are treated as open (inferred from those statements).

## Evidence
### Task: generation (causal language modeling)
- "We evaluate different Transformer variants with language modeling." (Abstract)
- "In this work, we focus on causal language modeling." (Limitations)
- "For every document, we select its first 4k tokens and divide them into the target length to fairly compare the perplexity of different lengths." (4.2 Language Modeling)
- "A Transformer model with a suitable design should be capable of dealing with any input length." (2.3 Length Extrapolation)
- "Our language model is trained on shorter texts in the same way as vanilla Transformers, i.e., using causal masking. During inference, we use blockwise causal attention for longer sequences, which recurrently reuses the overlapped parts (i.e., key and value vectors)." (Figure 2)
- Inference: Inferred 1D (t) input/output and token-generation output because the task is described as language modeling over "tokens" and evaluated by perplexity on token sequences; inferred Open dynamics from the claim that the model can handle "any input length" and the use of blockwise causal attention for longer sequences; inferred Static attention from the fixed causal/blockwise masking; inferred Direct state because the model is described as a standard causal language model with no explicit constructed state beyond the input.
