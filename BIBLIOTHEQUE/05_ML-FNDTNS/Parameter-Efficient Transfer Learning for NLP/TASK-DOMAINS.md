# Parameter-Efficient Transfer Learning for NLP (2019)
Source: Parameter-Efficient Transfer Learning for NLP.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text classification | text sequences (tokens) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class label | 0D (inferred) | Not specified in the paper. |
| extractive question answering | question and Wikipedia paragraph (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer span from paragraph (text span) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates adapter-based transfer on text classification tasks (GLUE and additional datasets) and on SQuAD extractive question answering. Inputs are text sequences (questions/paragraphs for QA) with 1D token order, while outputs are class labels (0D) or answer spans (1D), inferred from the described prediction setup. The OCR text does not specify input size dynamics or any dynamic attention/state mechanisms, so those dimensions remain unspecified.

## Evidence
### Task: text classification
- "we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark." (Abstract)
- "The first token in each sequence is a special \"classification token\"." (Section 3.1 Experimental Settings)
- "We attach a linear layer to the embedding of this token to predict the class label." (Section 3.1 Experimental Settings)
- Inference: Input is a token sequence (1D (t)) and the output is a class label (0D), inferred from the sequence/token description and the class-label prediction statement. (Section 3.1 Experimental Settings)

### Task: extractive question answering
- "Finally, we confirm that adapters work on tasks other than classification by running on SQuAD v1.1." (Section 3.5 SQuAD Extractive Question Answering)
- "Given a question and Wikipedia paragraph, this task requires selecting the answer span to the question from the paragraph." (Section 3.5 SQuAD Extractive Question Answering)
- Inference: Input is a question plus paragraph as token sequences (1D (t)) and output is an answer span (1D (t)), inferred from the described question/paragraph and answer-span selection. (Section 3.5 SQuAD Extractive Question Answering)
