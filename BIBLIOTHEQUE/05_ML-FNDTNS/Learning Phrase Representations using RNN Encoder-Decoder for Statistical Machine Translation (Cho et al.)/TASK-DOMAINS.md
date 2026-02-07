# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014)
Source: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (Cho et al.).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (machine translation) | source sequence of symbols (source phrase tokens) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | target sequence of symbols (target phrase tokens) | 1D (t) (inferred) | Open (inferred) |
| scoring (conditional probability of target given source) | pair of sequences (source and target phrases) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | probability score for sequence pair | 0D (inferred) | Fixed (inferred) |

## Summary
The paper centers on sequence-to-sequence machine translation, where a source word sequence is encoded and a target word sequence is generated. It also uses the same model to score source-target phrase pairs via conditional probabilities. The inputs and outputs are 1D sequences with variable length, while attention is static and the model constructs internal state via a fixed-length summary vector (inferred from the encoder-decoder description). Overall, the task coverage is confined to text sequences and their scalar probability scores.

## Evidence
### Task: generation (machine translation)
- "encodes a sequence of symbols into a fixedlength vector representation, and the other decodes the representation into another sequence of symbols." (Abstract)
- "The proposed RNN Encoder—Decoder with a novel hidden unit is empirically evaluated on the task of translating from English to French." (Section 1 Introduction)
- "One way is to use the model to generate a target sequence given an input sequence." (Section 2.2 RNN Encoder-Decoder)
- Inference: Classified inputs/outputs as 1D (t) with Open dynamics, attention as Static, and state as Constructed because the model maps variable-length sequences into a fixed-length summary vector and decodes sequentially from that summary. (Section 2.2 RNN Encoder-Decoder: "The encoder maps a variable-length source sequence to a fixed-length vector, and the decoder maps the vector representation back to a variable-length target sequence."; "the hidden state of the RNN is a summary c of the whole input sequence.")

### Task: scoring (conditional probability of target given source)
- "the model can be used to *score* a given pair of input and output sequences" (Section 2.2 RNN Encoder-Decoder)
- "train the RNN Encoder–Decoder (see Sec. 2.2) on a table of phrase pairs and use its scores as additional features" (Section 3.1 Scoring Phrase Pairs with RNN Encoder–Decoder)
- Inference: Classified inputs as 1D (t) with Open dynamics, attention as Static, and state as Constructed based on the variable-length sequence encoder and fixed-length summary vector used for scoring; output is a scalar probability score (0D, Fixed). (Section 2.2 RNN Encoder-Decoder: "The encoder maps a variable-length source sequence to a fixed-length vector, and the decoder maps the vector representation back to a variable-length target sequence."; "the hidden state of the RNN is a summary c of the whole input sequence.")
