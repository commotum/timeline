# DEBERTA: DECODING-ENHANCED BERT WITH DIS-ENTANGLED ATTENTION (Not specified in the paper.)
Source: DeBERTa- Decoding-enhanced BERT with Disentangled Attention.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked language modeling (MLM) | text tokens with masks (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | masked tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Replaced token detection (RTD) | text tokens with replacements (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | token-level replaced/real labels (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Auto-regressive language modeling (ARLM) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | next-token sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Acceptability (CoLA) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Sentiment (SST) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Natural language inference (MNLI) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Natural language inference (RTE) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Paraphrase (QQP) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Paraphrase (MRPC) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| QA/NLI (QNLI) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Similarity (STS-B) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | similarity score (inferred) | 0D (inferred) | Fixed (inferred) |
| MRC (SQuAD v1.1) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer span tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| MRC (SQuAD v2.0) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer span tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| MRC (ReCoRD) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer span tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| MRC (RACE) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | choice label (inferred) | 0D (inferred) | Fixed (inferred) |
| Multiple choice (SWAG) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | choice label (inferred) | 0D (inferred) | Fixed (inferred) |
| NER (CoNLL 2003) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | token-level entity labels (inferred) | 1D (t) (inferred) | Capped (inferred) |
| QA (BoolQ) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| QA (COPA) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | choice label (inferred) | 0D (inferred) | Fixed (inferred) |
| NLI (CB) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Multiple choice (MultiRC) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | choice label (inferred) | 0D (inferred) | Fixed (inferred) |
| WSD (WiC) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Coreference (WSC) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers text-only tasks spanning masked language modeling and replaced-token detection in pre-training, auto-regressive language modeling for generation, and a broad set of NLU benchmarks (GLUE/SuperGLUE, QA/MRC, and NER). Across tasks, inputs are treated as 1D token sequences with capped lengths and outputs are either 0D labels/scores or 1D token sequences/spans, all inferred from the sequence framing and task tables. Attention is described as Transformer self-attention and is therefore treated as Static (inferred), and state is treated as Direct (inferred).

## Evidence
### Task: Masked language modeling (MLM)
- "known as Masked Language Model (MLM) (Devlin et al., 2019)." (Section 2.2 Masked Language Model)
- "train a language model parameterized by  $\theta$  to reconstruct X by predicting the masked tokens" (Section 2.2 Masked Language Model)
- Inference: In/Out dimension, dynamics, attention, and state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Replaced token detection (RTD)
- "Replaced token detection (RTD) is a new pre-training objective introduced by ELECTRA (Clark et al., 2020)." (A.11 Further Improve the Model Efficiency)
- "we replace the MLM objective with the RTD objective" (A.11 Further Improve the Model Efficiency)
- Inference: In/Out dimension, dynamics, attention, and state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Auto-regressive language modeling (ARLM)
- "We evaluate DeBERTa on the task of auto-regressive language model (ARLM) using Wikitext-103 (Merity et al., 2016)." (A.4 Main Results on Generation Tasks)
- Inference: In/Out dimension, dynamics, attention, and state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Acceptability (CoLA)
- "| CoLA       | Acceptability   | 8.5k      | 1k        | 1k        | 2      | Matthews corr         |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Sentiment (SST)
- "| SST        | Sentiment       | 67k       | 872       | 1.8k      | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Natural language inference (MNLI)
- "| MNLI       | NLI             | 393k      | 20k       | 20k       | 3      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Natural language inference (RTE)
- "| RTE        | NLI             | 2.5k      | 276       | 3k        | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Paraphrase (QQP)
- "| QQP        | Paraphrase      | 364k      | 40k       | 391k      | 2      | Accuracy/F1           |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Paraphrase (MRPC)
- "| MRPC       | Paraphrase      | 3.7k      | 408       | 1.7k      | 2      | Accuracy/F1           |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: QA/NLI (QNLI)
- "| QNLI       | QA/NLI          | 108k      | 5.7k      | 5.7k      | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Similarity (STS-B)
- "| STS-B      | Similarity      | 7k        | 1.5k      | 1.4k      | 1      | Pearson/Spearman corr |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: MRC (SQuAD v1.1)
- "| SQuAD v1.1 | MRC             | 87.6k     | 10.5k     | 9.5k      | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: MRC (SQuAD v2.0)
- "| SQuAD v2.0 | MRC             | 130.3k    | 11.9k     | 8.9k      | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: MRC (ReCoRD)
- "| ReCoRD     | MRC             | 101k      | 10k       | 10k       | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: MRC (RACE)
- "| RACE       | MRC             | 87,866    | 4,887     | 4,934     | 4      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Multiple choice (SWAG)
- "| SWAG       | Multiple choice | 73.5k     | 20k       | 20k       | 4      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: NER (CoNLL 2003)
- "| CoNLL 2003 | NER             | 14,987    | 3,466     | 3,684     | 8      | F1                    |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: QA (BoolQ)
- "| BoolQ      | QA              | 9,427     | 3,270     | 3,245     | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: QA (COPA)
- "| COPA       | QA              | 400k      | 100       | 500       | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: NLI (CB)
- "| СВ         | NLI             | 250       | 57        | 250       | 3      | Accuracy/F1           |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Multiple choice (MultiRC)
- "| MultiRC    | Multiple choice | 5,100     | 953       | 1,800     | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: WSD (WiC)
- "| WiC        | WSD             | 2.5k      | 276       | 3k        | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).
### Task: Coreference (WSC)
- "| WSC        | Coreference     | 554k      | 104       | 146       | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Inference: Output object and 0D/1D structure are inferred from the task row (#Label/EM/F1), and sequence-based dynamics/attention/state are inferred from "given a sequence  $X = \{x_i\}$", "Each block contains a multi-head self-attention layer", and "the maximum sequence length that can be handled is 24,528" (Sections 2.1, 2.2, A.5).

