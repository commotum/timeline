# Addressing the Rare Word Problem in Neural Machine Translation (Not specified in the paper)
Source: Addressing the Rare Word Problem in Neural Machine Translation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation (generation) | Source sentence tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Target sentence tokens with OOV position tokens | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper describes a neural machine translation system that maps source-language sentences to target-language sentences, with outputs that include special tokens to track OOV word positions for post-processing. The task operates over token sequences (1D) and the training data caps sentence length at 100 tokens. The described model uses an encoder-decoder LSTM with a fixed sentence representation, implying static attention and constructed internal state.

## Evidence
### Task: Machine translation (generation)
- "reads the entire source sentence and produces an output translation one word at a time." (Section 1 Introduction)
- "maps a source sentence, s1,...,sn, to a target sentence, t1,...,tm" (Section 2 Neural Machine Translation)
- "emit, for each OOV word in the target sentence, the position of its corresponding word in the source sentence." (Abstract)
- "We discard sentence pairs in which the source or the target sentence exceed 100 tokens." (Section 4.1 Training Data)
- Inference: Labeled 1D (t) and Capped because inputs/outputs are sentence token sequences and they "discard sentence pairs in which the source or the target sentence exceed 100 tokens." (Section 4.1 Training Data). Labeled Attention Dynamic as Static and State Dynamic as Constructed because the encoder "produces a large vector that represents the entire source sentence" to initialize the decoder (Section 2 Neural Machine Translation).
