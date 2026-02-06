# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) (2020)
Source: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentence acceptability judgment (classification) | text (sentence) | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | label (single word) | 0D | Fixed |
| Sentiment analysis (classification) | text (sentence) | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | label (single word) | 0D | Fixed |
| Paraphrasing / sentence similarity (classification) | text (sentence pair) | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | label or similarity score (single word) | 0D | Fixed |
| Natural language inference (classification) | text (premise + hypothesis) | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | label (single word) | 0D | Fixed |
| Coreference/pronoun resolution (referent prediction) | text passage with ambiguous pronoun | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | referent noun phrase (text) | 1D (t) | Not specified in the paper. |
| Sentence completion (classification) | text | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | label (single word) | 0D | Fixed |
| Word sense disambiguation (classification) | text | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | label (single word) | 0D | Fixed |
| Question answering (reading comprehension) | question + context text | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) | Not specified in the paper. |
| Abstractive summarization | document text | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | summary text | 1D (t) | Not specified in the paper. |
| Machine translation (English->German) | English text | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | German text | 1D (t) | Not specified in the paper. |
| Machine translation (English->French) | English text | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | French text | 1D (t) | Not specified in the paper. |
| Machine translation (English->Romanian) | English text | 1D (t) | Capped | Not specified in the paper. | Not specified in the paper. | Romanian text | 1D (t) | Not specified in the paper. |

## Summary
The paper evaluates a text-to-text model across a wide range of NLP tasks, spanning text classification (GLUE/SuperGLUE categories), question answering, abstractive summarization, and machine translation. All tasks operate on text sequences (1D), with inputs capped by a maximum sequence length of 512 tokens, while outputs are either fixed single-word labels/scores (0D) or text sequences (1D) for generative tasks (output length not specified). The paper does not explicitly specify attention dynamics or state construction for the task definitions, so those remain unspecified here.

## Evidence
### Task: Sentence acceptability judgment (classification)
- "Sentence acceptability judgment (CoLA (Warstadt et al., 2018))" (Section 2.3)
- "For text classification tasks, the model simply predicts a single word corresponding to the target label." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Sentiment analysis (classification)
- "Sentiment analysis (SST-2 (Socher et al., 2013))" (Section 2.3)
- "For text classification tasks, the model simply predicts a single word corresponding to the target label." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Paraphrasing / sentence similarity (classification)
- "Paraphrasing/sentence similarity (MRPC (Dolan and Brockett, 2005), STS-B (Cer et al., 2017), QQP (Iyer et al., 2017))" (Section 2.3)
- "STS-B, which is a regression task where the goal is to predict a similarity score between 1 and 5." (Section 2.4)
- "This effectively recasts the STS-B regression problem as a 21-class classification problem." (Section 2.4)
- "For text classification tasks, the model simply predicts a single word corresponding to the target label." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Natural language inference (classification)
- "Natural language inference (MNLI (Williams et al., 2017), QNLI (Rajpurkar et al., 2016), RTE (Dagan et al., 2005), CB (De Marneff et al., 2019))" (Section 2.3)
- "For text classification tasks, the model simply predicts a single word corresponding to the target label." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Coreference/pronoun resolution (referent prediction)
- "Coreference resolution (WNLI and WSC (Levesque et al., 2012))" (Section 2.3)
- "we also include the Definite Pronoun Resolution (DPR) data set (Rahman and Ng, 2012) in the combined SuperGLUE task." (Section 2.3)
- "asking the model to predict the noun that it refers to." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Sentence completion (classification)
- "Sentence completion (COPA (Roemmele et al., 2011))" (Section 2.3)
- "For text classification tasks, the model simply predicts a single word corresponding to the target label." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Word sense disambiguation (classification)
- "Word sense disambiguation (WIC (Pilehvar and Camacho-Collados, 2018))" (Section 2.3)
- "For text classification tasks, the model simply predicts a single word corresponding to the target label." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Question answering (reading comprehension)
- "Question answering (MultiRC (Khashabi et al., 2018), ReCoRD (Zhang et al., 2018), BoolQ (Clark et al., 2019))" (Section 2.3)
- "SQuAD (Rajpurkar et al., 2016) is a common question-answering benchmark." (Section 2.3)
- "the model is fed the question and its context and asked to generate the answer token-by-token." (Section 2.3)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Abstractive summarization
- "The CNN/Daily Mail (Hermann et al., 2015) data set was introduced as a question-answering task but was adapted for text summarization" (Section 2.3)
- "we use the non-anonymized version from See et al. (2017) as an abstractive summarization task." (Section 2.3)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Machine translation (English->German)
- "WMT English to German, French, and Romanian translation." (Section 2.3)
- "translate English to German: That is good." (Section 2.4)
- "Das ist gut." (Section 2.4)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Machine translation (English->French)
- "WMT English to German, French, and Romanian translation." (Section 2.3)
- "We use a maximum sequence length of 512" (Section 3.1.2)

### Task: Machine translation (English->Romanian)
- "WMT English to German, French, and Romanian translation." (Section 2.3)
- "We use a maximum sequence length of 512" (Section 3.1.2)
