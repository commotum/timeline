# Attention Is All You Need (Not specified in the paper.)
Source: Attention Is All You Need.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation (English-to-German, English-to-French) | tokens (source sentences) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | tokens (target sentences) | 1D (t) (inferred) | Capped |
| English constituency parsing | sentences (tokens) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | constituency parses | 1D (t) (inferred) | Capped |

## Summary
The paper evaluates the Transformer on text sequence transduction tasks: machine translation (English-to-German and English-to-French) and English constituency parsing. Inputs and outputs are sequences of symbols/sentences, so the tasks align with 1D (t) domains, with variable-length inputs and capped output lengths during inference. Based on the described architecture, attention operates over all positions in the provided sequences and the encoder constructs intermediate sequence representations before decoding.

## Evidence
### Task: Machine translation (English-to-German, English-to-French)
- "Experiments on two machine translation tasks show these models to be superior in quality." (Abstract)
- "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task." (Abstract)
- "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0." (Section 6.1 Machine Translation)
- "sentence pairs containing approximately 25000 source tokens and 25000 target tokens." (Section 5.1 Training Data and Batching)
- "We set the maximum output length during inference to input length + 50." (Section 6.1 Machine Translation)
- "the encoder maps an input sequence of symbol representations (x1,...,xn) to a sequence of continuous representations z." (Section 3 Model Architecture)
- "the decoder then generates an output sequence (y1,...,ym) of symbols one element at a time." (Section 3 Model Architecture)
- "each position in the encoder can attend to all positions in the previous layer of the encoder." (Section 3.2.3 Applications of Attention in our Model)
- "We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training." (Section 3.5 Positional Encoding)
- Inference: In/Out Dimension = 1D (t), In Dynamics = Open, Attention Dynamic = Static, State Dynamic = Constructed based on the sequence-to-sequence architecture, length extrapolation note, and full-position self-attention described above.

### Task: English constituency parsing
- "we performed experiments on English constituency parsing." (Section 6.3 English Constituency Parsing)
- "on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences." (Section 6.3 English Constituency Parsing)
- "the output is subject to strong structural constraints and is significantly longer than the input." (Section 6.3 English Constituency Parsing)
- "During inference, we increased the maximum output length to input length + 300." (Section 6.3 English Constituency Parsing)
- "the encoder maps an input sequence of symbol representations (x1,...,xn) to a sequence of continuous representations z." (Section 3 Model Architecture)
- "the decoder then generates an output sequence (y1,...,ym) of symbols one element at a time." (Section 3 Model Architecture)
- "each position in the encoder can attend to all positions in the previous layer of the encoder." (Section 3.2.3 Applications of Attention in our Model)
- "We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training." (Section 3.5 Positional Encoding)
- Inference: In/Out Dimension = 1D (t), In Dynamics = Open, Attention Dynamic = Static, State Dynamic = Constructed based on the sequence-to-sequence architecture, length extrapolation note, and full-position self-attention described above.
