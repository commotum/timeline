# Nystromformer: A Nystrom-based Algorithm for Approximating Self-Attention (2021)
Source: Nystromformer- A Nystrom-Based Algorithm for Approximating Self-Attention.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked language modeling | tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | tokens (masked positions) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Sentence-order prediction | tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (SST-2) (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (MRPC) (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (QNLI) (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (QQP) (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (MNLI) (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (IMDB reviews) (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (Listops) (inferred) | sequence (Listops) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (byte-level IMDb reviews text classification) | byte-level text (IMDb reviews) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Retrieval (byte-level document retrieval) | byte-level documents (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (image classification on sequences of pixels) | sequences of pixels | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| Classification (Pathfinder) (inferred) | sequence (Pathfinder) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates Nystromformer on pretraining objectives (masked language modeling and sentence-order prediction), GLUE/IMDB downstream NLP datasets, and Long Range Arena tasks including Listops, byte-level text classification, document retrieval, pixel-sequence image classification, and Pathfinder. The tasks are primarily sequence-based inputs with capped lengths in the downstream and LRA settings, producing mostly 0D label outputs; MLM is the only token-level output task. Attention and state dynamics are inferred as static and direct because the model applies Transformer self-attention over the provided input sequences.

## Evidence
### Task: Masked language modeling
- "maskedlanguage-modeling" (Section: (Pre-)training of Language Modeling)
- Inference: Input/output tokens, 1D structure, and static/direct attention/state are inferred from the self-attention description over an input token sequence; MLM implies predicting masked tokens. (Section: Self-Attention; Section: (Pre-)training of Language Modeling)

### Task: Sentence-order prediction
- "sentence-order-prediction" (Section: (Pre-)training of Language Modeling)
- Inference: Input tokens and 1D structure are inferred from the self-attention description over an input token sequence; output labels and fixed 0D outputs are inferred from the prediction objective. (Section: Self-Attention; Section: (Pre-)training of Language Modeling)

### Task: Classification (SST-2) (inferred)
- "SST-2" (Section: Fine-tuning on Downstream NLP tasks)
- Inference: Classification/label outputs are inferred from Table 2 reporting F1/accuracy for these datasets; inputs are inferred as token sequences; capped input length is inferred from the stated maximum input sequence length 512 for downstream tasks. (Section: Fine-tuning on Downstream NLP tasks)

### Task: Classification (MRPC) (inferred)
- "MRPC" (Section: Fine-tuning on Downstream NLP tasks)
- Inference: Classification/label outputs are inferred from Table 2 reporting F1/accuracy for these datasets; inputs are inferred as token sequences; capped input length is inferred from the stated maximum input sequence length 512 for downstream tasks. (Section: Fine-tuning on Downstream NLP tasks)

### Task: Classification (QNLI) (inferred)
- "QNLI" (Section: Fine-tuning on Downstream NLP tasks)
- Inference: Classification/label outputs are inferred from Table 2 reporting F1/accuracy for these datasets; inputs are inferred as token sequences; capped input length is inferred from the stated maximum input sequence length 512 for downstream tasks. (Section: Fine-tuning on Downstream NLP tasks)

### Task: Classification (QQP) (inferred)
- "QQP" (Section: Fine-tuning on Downstream NLP tasks)
- Inference: Classification/label outputs are inferred from Table 2 reporting F1/accuracy for these datasets; inputs are inferred as token sequences; capped input length is inferred from the stated maximum input sequence length 512 for downstream tasks. (Section: Fine-tuning on Downstream NLP tasks)

### Task: Classification (MNLI) (inferred)
- "MNLI" (Section: Fine-tuning on Downstream NLP tasks)
- Inference: Classification/label outputs are inferred from Table 2 reporting F1/accuracy for these datasets; inputs are inferred as token sequences; capped input length is inferred from the stated maximum input sequence length 512 for downstream tasks. (Section: Fine-tuning on Downstream NLP tasks)

### Task: Classification (IMDB reviews) (inferred)
- "IMDB" (Section: Fine-tuning on Downstream NLP tasks)
- Inference: Classification/label outputs are inferred from Table 2 reporting F1/accuracy for these datasets; inputs are inferred as token sequences; capped input length is inferred from the stated maximum input sequence length 512 for downstream tasks. (Section: Fine-tuning on Downstream NLP tasks)

### Task: Classification (Listops) (inferred)
- "Listops" (Section: Long Range Arena (LRA) Benchmark)
- Inference: Classification/label outputs are inferred from the LRA section reporting classification accuracy for each task; input sequence structure and capped dynamics are inferred from the ListOps (2K) sequence length in Table 3. (Section: Long Range Arena (LRA) Benchmark; Table 3)

### Task: Classification (byte-level IMDb reviews text classification)
- "IMDb" (Section: Long Range Arena (LRA) Benchmark)
- Inference: Byte-level text inputs and 1D structure are inferred from the byte-level task description; label outputs and capped dynamics are inferred from the LRA classification-accuracy reporting and the Text (4K) length in Table 3. (Section: Long Range Arena (LRA) Benchmark; Table 3)

### Task: Retrieval (byte-level document retrieval)
- "retrieval" (Section: Long Range Arena (LRA) Benchmark)
- Inference: Byte-level document inputs and 1D structure are inferred from the task description; label outputs and capped dynamics are inferred from the LRA classification-accuracy reporting and the Retrieval (4K) length in Table 3. (Section: Long Range Arena (LRA) Benchmark; Table 3)

### Task: Classification (image classification on sequences of pixels)
- "image" (Section: Long Range Arena (LRA) Benchmark)
- Inference: 1D sequence input and capped dynamics are inferred from the sequences of pixels description and the Image (1K) length in Table 3; label outputs are inferred from the LRA classification-accuracy reporting. (Section: Long Range Arena (LRA) Benchmark; Table 3)

### Task: Classification (Pathfinder) (inferred)
- "Pathfinder" (Section: Long Range Arena (LRA) Benchmark)
- Inference: Classification/label outputs are inferred from the LRA section reporting classification accuracy for each task; input sequence structure and capped dynamics are inferred from the Pathfinder (1K) length in Table 3. (Section: Long Range Arena (LRA) Benchmark; Table 3)
