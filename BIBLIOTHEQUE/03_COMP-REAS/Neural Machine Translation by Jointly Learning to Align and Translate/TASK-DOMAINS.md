# Neural Machine Translation by Jointly Learning to Align and Translate (2014)
Source: Neural Machine Translation by Jointly Learning to Align and Translate.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation (English-to-French) | Source sentence word tokens (English) | 1D (t) (inferred) | Open (inferred) | Dynamic | Constructed (inferred) | Target sentence word tokens (French) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper addresses neural machine translation, specifically English-to-French sentence translation, producing target-language sentences from source sentences. Inputs and outputs are word-token sequences, implying 1D (t) structures with open length variability, and the decoder uses a dynamic attention mechanism over source positions. The model maintains internal hidden states and context vectors, indicating constructed state.

## Evidence
### Task: Machine translation (English-to-French)
- "neural machine translation attempts to build and train a single, large neural network that reads a sentence and outputs a correct translation." (Section 1 Introduction)
- "We evaluate the proposed approach on the task of English-to-French translation." (Section 4 Experiment Settings)
- "The model takes a source sentence of 1-of-K coded word vectors as input" (Section A.2.1 Encoder)
- "and outputs a translated sentence of 1-of-K coded word vectors" (Section A.2.1 Encoder)
- "encode a variable-length source sentence into a fixed-length vector and to decode the vector into a variable-length target sentence." (Section 2.1 RNN ENCODER-DECODER)
- "The decoder decides parts of the source sentence to pay attention to." (Section 3.1 Decoder: General Description)
- "s_i is an RNN hidden state for time i" (Section 3.1 Decoder: General Description)
- "The context vector c_i is, then, computed as a weighted sum of these annotations h_i." (Section 3.1 Decoder: General Description)
- Inference: In/Out Dimension are 1D (t) and In/Out Dynamics are Open because inputs/outputs are sequences indexed by position (x1..T_x, y1..T_y) and described as variable-length sentences; State Dynamic is Constructed because the model uses decoder hidden states and context vectors beyond the raw input. (See quotes on sequence inputs/outputs, variable-length sentences, and hidden state.)
