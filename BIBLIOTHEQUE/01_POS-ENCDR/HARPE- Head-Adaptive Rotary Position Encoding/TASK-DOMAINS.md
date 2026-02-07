# Breaking the Stage Barrier: A Novel Single-Stage Approach to Long Context Extension for Large Language Models (Not specified in the paper.)
Source: HARPE- Head-Adaptive Rotary Position Encoding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Perplexity evaluation (Proof-pile) | Proof-pile documents (token sequences) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | perplexity score (PPL) | Not specified in the paper. | Not specified in the paper. |
| Perplexity evaluation (GovReport) | GovReport documents (token sequences) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | perplexity score (PPL) | Not specified in the paper. | Not specified in the paper. |
| Needle-in-a-Haystack retrieval/recitation (multi-key, multi-value, multi-query) | lengthy document (haystack) with needle sentence(s) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | specific sentence (needle) | 1D (t) (inferred) | Not specified in the paper. |
| RULER benchmark (13 long-context tasks) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| MMLU (5-shot) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Hellaswag (10-shot) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| ARC-Challenge (25-shot) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| PIQA (0-shot) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| TriviaQA (5-shot) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates HARPE on text-only NLP benchmarks: perplexity on Proof-pile and GovReport documents, Needle-in-a-Haystack variants, the RULER long-context benchmark, and short-context datasets (MMLU, Hellaswag, ARC-Challenge, PIQA, TriviaQA). Where specified, inputs are token sequences/documents with capped context lengths (up to 128k tokens), implying 1D (t) inputs; the NIAH task outputs specific sentences, also 1D (t). Attention dynamics and state dynamics are not specified, and outputs for most benchmarks are not described beyond task names.

## Evidence
### Task: Perplexity evaluation (Proof-pile)
- "Perplexity (PPL) is evaluated on the Proof-pile (Zhangir Azerbayev, 2022) and GovReport (Huang et al., 2021) datasets." (Section 4.3 Evaluation Metric)
- "Sliding window perplexity (S = 256) for **Proof**pile and GovReport documents." (Table 3 caption)
- Inference: Inferred 1D (t) input and Capped input dynamics because the evaluation uses "token lengths ranging from 2k to 128k in increments of 2k". (Section 4.3 Evaluation Metric)

### Task: Perplexity evaluation (GovReport)
- "Perplexity (PPL) is evaluated on the Proof-pile (Zhangir Azerbayev, 2022) and GovReport (Huang et al., 2021) datasets." (Section 4.3 Evaluation Metric)
- "Sliding window perplexity (S = 256) for **Proof**pile and GovReport documents." (Table 3 caption)
- Inference: Inferred 1D (t) input and Capped input dynamics from "context window of 32k tokens". (Section 4.3 Evaluation Metric)

### Task: Needle-in-a-Haystack retrieval/recitation (multi-key, multi-value, multi-query)
- "Needle-in-a-Haystack is a task that assesses a model's ability to accurately locate and recite a specific sentence" (Section 4.3 Evaluation Metric)
- "within a lengthy document, known as the \"haystack\"." (Section 4.3 Evaluation Metric)
- "to include multi-key, multi-value and multi-query scenarios" (Section 4.3 Evaluation Metric)
- Inference: Inferred 1D (t) input/output and Capped input dynamics because the task operates on a "lengthy document" and evaluation ranges "up to 128k tokens". (Section 4.3 Evaluation Metric; Figure 2 caption)

### Task: RULER benchmark (13 long-context tasks)
- "In this section, we evaluate HARPE against various open-source pre-trained models on a range of long-context tasks using the RULER benchmark." (Section 5.3 Comparative Results on RULER Evaluation)
- "RULER is a comprehensive and widely recognized standard for long-context evaluation" (Section 5.3 Comparative Results on RULER Evaluation)
- "13 tasks that include \"needle in a haystack\"" (Section 5.3 Comparative Results on RULER Evaluation)
- "additional tasks such as Variable Tracing, Aggregation Ability, and Question Answering." (Section 5.3 Comparative Results on RULER Evaluation)

### Task: MMLU (5-shot)
- "5-shot MMLU (Hendrycks et al., 2020)" (Section 4.3 Evaluation Metric)

### Task: Hellaswag (10-shot)
- "10-shot Hellaswag (Zellers et al., 2019)" (Section 4.3 Evaluation Metric)

### Task: ARC-Challenge (25-shot)
- "25-shot ARC-Challenge (Clark et al., 2018)" (Section 4.3 Evaluation Metric)

### Task: PIQA (0-shot)
- "0-shot PiQA (Bisk et al., 2019)" (Section 4.3 Evaluation Metric)

### Task: TriviaQA (5-shot)
- "5-shot TriviaQA (Joshi et al., 2017)" (Section 4.3 Evaluation Metric)
