# DEPTH-ADAPTIVE TRANSFORMER (Not specified in the paper)
Source: Depth-Adaptive Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| translation (sequence-to-sequence generation) | source sequence tokens (x) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | target sequence tokens (y) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates depth-adaptive Transformer models on machine translation, specifically German-English and English-French translation benchmarks, which are sequence-to-sequence generation tasks. Inputs and outputs are token sequences, implying 1D temporal structure, while the model uses standard Transformer self-/cross-attention (no runtime selection beyond the fixed sequences). The paper does not specify explicit interface bounds for input or output length, so dynamics are left unspecified.

## Evidence
### Task: translation (sequence-to-sequence generation)
- "On IWSLT German-English translation our approach matches the accuracy of a well tuned baseline Transformer while using less than a quarter of the decoder layers." (Abstract)
- "We encode the input sequence using a standard Transformer encoder to generate the output sequence with a varying amount of computation in the decoder network." (Introduction)
- "Given a pair of source-target sequences (x, y), x is processed with the encoder to give representations s = (s1, \dots, s|x|). Next, the decoder generates y step-by-step." (Anytime Structured Prediction - Transformer with Multiple Output Classifiers)
- "**IWSLT'14 German to English (De-En).**" (Experimental Setup)
- "WMT'14 English to French (En-Fr)." (Experimental Setup)
- Inference: In/Out Dimension set to "1D (t)" because the paper frames both input and output as sequences and describes token-by-token decoding; Attention Dynamic set to Static because the model uses standard Transformer self-/cross-attention over the provided sequences without runtime selection of external information; State Dynamic set to Direct because the task is next-token prediction over sequences without an explicit external state beyond the input tokens. Supporting text includes the sequence and token-by-token descriptions above. (Anytime Structured Prediction - Transformer with Multiple Output Classifiers; Introduction)
