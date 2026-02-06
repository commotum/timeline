# ENCODING WORD ORDER IN COMPLEX EMBEDDINGS (Not specified in the paper.)
Source: Encoding Word Order in Complex Embeddings.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text classification | text (word tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | class labels (inferred) | 0D (inferred) | Not specified in the paper. |
| machine translation | source-language sentences (tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | target-language sentences (tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| language modeling | text (character sequence) (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | characters (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates its embeddings on three text-only tasks: text classification, machine translation, and language modeling. Based on task descriptions and datasets, inputs are textual sequences (words or characters) with 1D (t) structure, producing class labels (0D) for classification and text sequences (1D) for translation and language modeling (inferred). The architecture descriptions include self-attention and recurrent hidden states, so attention is treated as Static and state as Constructed when inferred, but explicit Fixed/Capped/Open dynamics for inputs or outputs are not specified.

## Evidence
### Task: text classification
- "We evaluate our embeddings in text classification, machine translation and language modeling." (Section 3 EXPERIMENTAL EVALUATION)
- "We use six popular text classification datasets: CR, MPQA, SUBJ, MR, SST, and TREC (see Tab. 1)." (Section 3.1 Text Classification)
- "| Dataset                   | train | test | vocab. | task             | Classes |" (Table 1: Dataset Statistics)
- "We use Fasttext (Joulin et al., 2016), CNN (Kim, 2014), LSTM and Transformer (Vaswani et al., 2017) as NN baselines" (Section 3.1 Text Classification)
- "The main components in the Transformer are self-attention sublayers and position-wise feed-forward (FFN) sublayers." (Appendix B)
- "where  $z_t$  and  $h_t$  represent the complex-valued input and complex-value hidden state vectors at time t" (Appendix B)
- Inference: Text classification over word embeddings implies 1D token-sequence inputs and 0D class-label outputs; the use of Transformer self-attention over a fixed sequence and recurrent hidden states supports Static attention and Constructed state (based on the quotes above).

### Task: machine translation
- "We use the standard WMT 2016 English-German dataset (Sennrich et al., 2016), whose training set consists of 29,000 sentence pairs." (Section 3.2 MACHINE TRANSLATION)
- "and a 6-layer Transformer." (Section 3.2 MACHINE TRANSLATION)
- "The main components in the Transformer are self-attention sublayers and position-wise feed-forward (FFN) sublayers." (Appendix B)
- Inference: The mention of English-German sentence pairs implies a 1D sequence-to-sequence text task with source sentences as input and target sentences as output; Transformer self-attention over the fixed input sequence supports Static attention and Constructed state (based on the quotes above).

### Task: language modeling
- "We use the text8 (Mahoney, 2011) dataset, consisting of English Wikipedia articles." (Section 3.3 Language Modeling)
- "The text is lowercased from a to z, and space." (Section 3.3 Language Modeling)
- "We evaluate performance with the Bits Per Character (BPC) measure" (Section 3.3 Language Modeling)
- "our model, named Transformer XL complex-order, directly replaces the word embedding with our proposed embedding under the same setting." (Section 3.3 Language Modeling)
- "The main components in the Transformer are self-attention sublayers and position-wise feed-forward (FFN) sublayers." (Appendix B)
- Inference: The text8 character set and BPC metric indicate character-sequence modeling with 1D inputs and character outputs; Transformer XL usage implies Static attention over a fixed input sequence and a Constructed state (based on the quotes above).
