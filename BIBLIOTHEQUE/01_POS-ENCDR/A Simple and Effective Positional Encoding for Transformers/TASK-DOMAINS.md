# A Simple and Effective Positional Encoding for Transformers (Not specified in the paper)
Source: A Simple and Effective Positional Encoding for Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked language modeling | text tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | token predictions (masked positions) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| classification | text tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | class label (inferred) | 0D (inferred) | Fixed (inferred) |
| question answering | text tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer text span (inferred) | 1D (t) (inferred) | Capped (inferred) |
| machine translation | text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | translated text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates DIET on text-only NLP tasks spanning masked language modeling pre-training, text classification, question answering, and machine translation. All tasks operate over 1D token sequences, with capped input lengths for pre-training and finetuning tasks and unspecified sequence limits for translation. Attention is described via standard Transformer self-attention over sequence inputs, so attention is treated as static and state as direct (inferred).

## Evidence
### Task: masked language modeling
- "**Pre-training** We pre-train the models using a masked LM task (Devlin et al., 2018) and do not use the Next Sentence Prediction (NSP) loss as suggested in RoBERTa (Liu et al., 2019)." (Appendix A Experimental setup)
- "Each input is constructed with full sentences from documents, and packed up to the maximum sequence length." (Appendix A Experimental setup)
- "Given input sequence length n, hidden size d, multi-head query-key down-projection size  $d_h$ , we define hidden layer input to this attention head as  $\mathbf{X} \in \mathbb{R}^{n \times d}$" (Section 2.1 Transformer)
- Inference: Input/output dimensions marked 1D (t), dynamics marked Capped, attention marked Static, state marked Direct, and outputs marked as token predictions based on the masked LM task plus capped sequence length and the Transformer self-attention formulation over the full input sequence (Appendix A; Section 2.1).

### Task: classification
- "Second, we study zero-shot cross-lingual transferability of the multilingual pretrained models (Hu et al., 2020) to classification and question answering tasks in the XTREME benchmark (Hu et al., 2020)." (Section 4 Experiments)
- "**Classification** We conduct 5 trials of fine-tuning for each model on the MultiNLI (Williams et al., 2018) training data, then perform zero-shot predictions on XNLI (Conneau et al., 2018), choosing median accuracy to report." (Section 4.2 Cross-lingual Model Results)
- "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results)
- "Each input is constructed with full sentences from documents, and packed up to the maximum sequence length." (Appendix A Experimental setup)
- Inference: Input dimension marked 1D (t), input dynamics marked Capped, attention marked Static, state marked Direct, and output marked as a 0D class label based on the classification task framing, max sequence length, and standard Transformer self-attention over full input sequences (Section 4.2; Appendix A; Section 2.1).

### Task: question answering
- "Second, we study zero-shot cross-lingual transferability of the multilingual pretrained models (Hu et al., 2020) to classification and question answering tasks in the XTREME benchmark (Hu et al., 2020)." (Section 4 Experiments)
- "**Question Answering** We conduct 5 trials of finetuning for each model on SQuAD V1.1 dataset, following by zero-shot predictions on XQuAD (11 languages), MLQA (7 languages) and TyDiQA-GoldP (9 languages), choosing median F1 / EM scores to report." (Section 4.2 Cross-lingual Model Results)
- "Performance is measured by *accuracy* for classification, and *f1 score / exact match* for question answering." (Table 3: XTREME)
- "We use language-independent tokenizer, Sentence Piece (Kudo and Richardson, 2018) model, with 120,000 token vocabulary to encode input text." (Section 4.2 Cross-lingual Model Results)
- Inference: Input dimension marked 1D (t), input dynamics marked Capped, attention marked Static, state marked Direct, and output marked as an answer text span (1D) with capped dynamics based on the question answering setup, F1/EM evaluation, tokenized text inputs, and capped sequence length in the training setup (Section 4.2; Table 3; Appendix A).

### Task: machine translation
- "Lastly, we consider training Transformer models from scratch for machine translation." (Section 4 Experiments)
- "**Datasets and Model** For the machine translation task we consider two language pairs (both directions) for training - WMT 2018 English-to-German (en-de), German-to-English (de-en), English-to-Czech (en-cs) and Czech-to-English (cs-en) (Bojar et al., 2018)." (Section 4.3 Translation Results)
- "We train a 6 layer Transformer model." (Appendix A Experimental setup)
- "Any changes to position encoding are applied to all the attention layers both in the encoder and decoder." (Appendix A Experimental setup)
- Inference: Input/output marked as text token sequences with 1D (t) dimensions, attention marked Static, and state marked Direct based on the machine translation task framing and standard Transformer encoder-decoder attention; input/output dynamics are not specified for translation in the paper (Section 4.3; Appendix A).
