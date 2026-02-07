# NEZHA: NEURAL CONTEXTUALIZED REPRESENTATION FOR CHINESE LANGUAGE UNDERSTANDING (2021)
Source: NEZHA- Neural Contextualized Representation for Chinese Language Understanding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (masked token recovery / MLM) | training sentences with masked words | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | masked words (predicted tokens) | 1D (t) (inferred) | Not specified in the paper. |
| classification (next sentence prediction / NSP) | pair of sentences | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | whether one sentence is the next sentence of the other | 0D (inferred) | Not specified in the paper. |
| span prediction | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| span extraction (machine reading comprehension / CMRC) | passage and question | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer span in passage | 1D (t) (inferred) | Not specified in the paper. |
| classification (natural language inference / XNLI) | pair of sentences | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | contradiction/entailment/neutral label | 0D (inferred) | Not specified in the paper. |
| classification (sentence pair matching / LCQMC) | pair of sentences | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | semantic equivalence label | 0D (inferred) | Not specified in the paper. |
| sequence labeling (named entity recognition / PD-NER) | text | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | named entities from text | 1D (t) (inferred) | Not specified in the paper. |
| classification (sentiment / ChnSenti) | sentence | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | positive/negative sentiment label | 0D (inferred) | Not specified in the paper. |

## Summary
The paper describes NEZHA pre-training with MLM and NSP objectives and fine-tuning on five Chinese NLU tasks: CMRC span extraction, XNLI NLI, LCQMC sentence matching, PD-NER sequence labeling, and ChnSenti sentiment classification. All tasks operate on Chinese text (sentences, passages, or sentence pairs); based on the model’s token-sequence description, inputs are 1D (t) and outputs are 0D labels or 1D spans/labels (inferred). A span prediction objective is listed in the pre-training techniques table, but its input/output details are not specified. Task-level dynamics, attention dynamics, and state dynamics are not explicitly specified.

## Evidence
### Task: prediction (masked token recovery / MLM)
- "In the MLM task, the model learns to recover the masked words in the training sentences." (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- "In Transformer, each attention head operates on a sequence of tokens" (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- Inference: In Dimension = 1D (t) (inferred) and Out Dimension = 1D (t) (inferred) because the task operates on sentence token sequences and recovers masked words within them.

### Task: classification (next sentence prediction / NSP)
- "Each sample in the training data of BERT is a pair of sentences." (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- "In the NSP task, it tries to predict whether one sentence is the next sentence of the other." (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- "In Transformer, each attention head operates on a sequence of tokens" (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- Inference: In Dimension = 1D (t) (inferred) because the input is a sentence pair encoded as a token sequence; Out Dimension = 0D (inferred) because NSP predicts a single next/not-next label.

### Task: span prediction
- "Span Prediction" (Table 2: Pre-training Techniques Adopted in Chinese pre-trained language models)

### Task: span extraction (machine reading comprehension / CMRC)
- "A machine reading comprehension task that returns an answer span in a given passage for a given question." (Section 3.2 Experimental Results)
- "In Transformer, each attention head operates on a sequence of tokens" (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- Inference: In Dimension = 1D (t) (inferred) because the input is passage/question text; Out Dimension = 1D (t) (inferred) because the output is an answer span within the passage.

### Task: classification (natural language inference / XNLI)
- "The goal of this task is to predict if the second sentence is a contradiction, entailment or neutral to the first sentence." (Section 3.2 Experimental Results)
- "In Transformer, each attention head operates on a sequence of tokens" (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- Inference: In Dimension = 1D (t) (inferred) for the sentence-pair input; Out Dimension = 0D (inferred) because the output is a single NLI label.

### Task: classification (sentence pair matching / LCQMC)
- "A sentence pair matching task." (Section 3.2 Experimental Results)
- "Given a pair of sentences, the task is to determine if the two sentences are semantically equivalent or not." (Section 3.2 Experimental Results)
- "In Transformer, each attention head operates on a sequence of tokens" (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- Inference: In Dimension = 1D (t) (inferred) for the sentence-pair input; Out Dimension = 0D (inferred) because the output is a single equivalence label.

### Task: sequence labeling (named entity recognition / PD-NER)
- "A sequence labeling task that identifies the named entities from text." (Section 3.2 Experimental Results)
- "In Transformer, each attention head operates on a sequence of tokens" (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- Inference: In Dimension = 1D (t) (inferred) because the input is text; Out Dimension = 1D (t) (inferred) because the task outputs sequence labels/identified entities along the text.

### Task: classification (sentiment / ChnSenti)
- "A binary classification task which predicts if the sentiment of a given sentence is positive or negative." (Section 3.2 Experimental Results)
- "In Transformer, each attention head operates on a sequence of tokens" (Section 2.1 Preliminaries: BERT Model & Positional Encoding)
- Inference: In Dimension = 1D (t) (inferred) because the input is a sentence; Out Dimension = 0D (inferred) because the output is a single sentiment label.
