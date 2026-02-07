# Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding (Not specified in the paper)
Source: Length Extrapolation of Transformers- A Survey from the Perspective of Positional Encoding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language modeling | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| machine translation | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | translated text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| question answering | question + context text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| summarization | document text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | summary text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| code completion | code tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | code tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| arithmetic calculation (addition/subtraction) | addition/subtraction sequences | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | calculated result (inferred) | 0D (inferred) | Fixed (inferred) |
| deductive reasoning | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The survey frames length extrapolation around NLP generation and evaluation tasks, explicitly referencing language modeling, machine translation, question answering, summarization, and code completion, and also discusses synthetic arithmetic and deductive reasoning tasks. The only structural signals provided are that Transformers process sequences and are trained with maximum length limits, so the text-based tasks can be justified as 1D sequences with capped input lengths (inferred). Attention dynamics and state construction are not specified for these tasks in the paper.

## Evidence
### Task: language modeling
- "language modeling and perplexity have emerged as the standard metrics for evaluating length extrapolation" (Section 5 Evaluation and Benchmark)
- "Table 3: Empirical comparisons of different PEs on language modeling." (Section A.2 Results on Language Modeling)
- Inference: Treated input/output as 1D token sequences with capped input length because the paper defines Transformer input "as a sequence of n embeddings with dimension d" and notes models are "trained on sequences with a maximum length." (Section 2; Introduction)

### Task: machine translation
- "evaluation samples and metrics came from various downstream tasks such as machine translation and question answering." (Section 5 Evaluation and Benchmark)
- Inference: Interpreted machine translation as 1D text token sequences with capped input length based on "as a sequence of n embeddings with dimension d" and "trained on sequences with a maximum length." (Section 2; Introduction)

### Task: question answering
- "evaluation samples and metrics came from various downstream tasks such as machine translation and question answering." (Section 5 Evaluation and Benchmark)
- "| QA              |                          |                     |                  |                    |" (Section A.1 Length Extrapolation on Generation Tasks, Table 2)
- Inference: Interpreted question answering as 1D text token sequences with capped input length based on "as a sequence of n embeddings with dimension d" and "trained on sequences with a maximum length." (Section 2; Introduction)

### Task: summarization
- "| Summarization   |                          |                     |                  |                    |" (Section A.1 Length Extrapolation on Generation Tasks, Table 2)
- Inference: Interpreted summarization as 1D text token sequences with capped input length based on "as a sequence of n embeddings with dimension d" and "trained on sequences with a maximum length." (Section 2; Introduction)

### Task: code completion
- "| Code Completion |                          |                     |                  |                    |" (Section A.1 Length Extrapolation on Generation Tasks, Table 2)
- Inference: Treated code completion as 1D code token sequences with capped input length based on the task label "Code Completion" and the paper's sequence framing "as a sequence of n embeddings with dimension d" plus the "maximum length" constraint. (Section A.1; Section 2; Introduction)

### Task: arithmetic calculation (addition/subtraction)
- "synthetic tasks such as arithmetic and deductive reasoning in a controlled setup" (Section 6.2 Length Extrapolation and Generalization)
- "calculating long sequences containing only addition and subtraction within ten (and keeping the intermediate results in a small range)" (Section A.3 Thoughts on Standardized Benchmark)
- Inference: Treated inputs as 1D sequences and outputs as a single numeric result because the task is described as "calculating long sequences containing only addition and subtraction" and the paper defines Transformer input "as a sequence of n embeddings with dimension d." (Section A.3; Section 2)

### Task: deductive reasoning
- "synthetic tasks such as arithmetic and deductive reasoning in a controlled setup" (Section 6.2 Length Extrapolation and Generalization)
