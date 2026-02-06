# Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings (Not specified in the paper)
Source: Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (next-token prediction / perplexity) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | next-token predictions (text tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Needle-in-a-haystack retrieval (standard) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | retrieved values (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Needle-in-a-haystack retrieval (multi-query) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | retrieved values (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Needle-in-a-haystack retrieval (multi-key) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | retrieved values (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Needle-in-a-haystack retrieval (multi-value) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | retrieved values (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Long-context language modeling task (MultiFieldQA) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | task response (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Long-context language modeling task (MuSiQue) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | task response (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Long-context language modeling task (GovReport) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | task response (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Long-context language modeling task (LCC) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | task response (text tokens) (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Multiple-choice science QA (ARC-E) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | choice label (selected option) (inferred) | 0D (inferred) | Fixed (inferred) |
| Multiple-choice science QA (ARC-C) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | choice label (selected option) (inferred) | 0D (inferred) | Fixed (inferred) |
| Multiple-choice sentence completion (HellaSwag) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | choice label (selected option) (inferred) | 0D (inferred) | Fixed (inferred) |
| Multiple-choice open-book QA (OpenBookQA) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | choice label (selected option) (inferred) | 0D (inferred) | Fixed (inferred) |
| Multiple-choice physical commonsense reasoning (PIQA) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | choice label (selected option) (inferred) | 0D (inferred) | Fixed (inferred) |
| Multiple-choice coreference/commonsense (WinoGrande) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | choice label (selected option) (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates a language model on text-only tasks including language modeling (perplexity), long-context retrieval (needle-in-a-haystack variants), LongBench tasks (MultiFieldQA, MuSiQue, GovReport, LCC), and multiple-choice reasoning/QA benchmarks (ARC-E/C, HellaSwag, OpenBookQA, PIQA, WinoGrande). Inputs are token sequences with a capped context length (C_train/C_test and fixed maximum context length are specified), and outputs are either token sequences for generation/retrieval or single-choice labels for multiple-choice benchmarks. Attention and state dynamics are not explicitly specified in the task descriptions.

## Evidence
### Task: Language modeling (next-token prediction / perplexity)
- "DroPE matches RoPE's in-context perplexity." (Figure 2)
- "NoPE transformers maintain visibly worse perplexity throughout training." (Section 3)
- Inference: Input/Output treated as token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}" (Section 2).

### Task: Needle-in-a-haystack retrieval (standard)
- "We evaluate long-context retrieval using the needle-in-a-haystack (NIAH) setup." (Section C.2. Evaluation)
- "(Standard NIAH) We insert a single needle and prompt the model to retrieve it." (Section C.2. Evaluation)
- Inference: Input/Output treated as token sequences (1D (t)) with capped context length based on "h_1, \ldots, h_T" and "fixed maximum context length" (Sections 2, C.2).

### Task: Needle-in-a-haystack retrieval (multi-query)
- "Multi-Query NIAH: We insert multiple (key, value) pairs and prompt the model to return as many values as possible" (Section C.2. Evaluation)
- "We evaluate long-context retrieval using the needle-in-a-haystack (NIAH) setup." (Section C.2. Evaluation)
- Inference: Input/Output treated as token sequences (1D (t)) with capped context length based on "h_1, \ldots, h_T" and "fixed maximum context length" (Sections 2, C.2).

### Task: Needle-in-a-haystack retrieval (multi-key)
- "(Multi-Key NIAH) We insert multiple (key, value) pairs but query for a single key" (Section C.2. Evaluation)
- "We evaluate long-context retrieval using the needle-in-a-haystack (NIAH) setup." (Section C.2. Evaluation)
- Inference: Input/Output treated as token sequences (1D (t)) with capped context length based on "h_1, \ldots, h_T" and "fixed maximum context length" (Sections 2, C.2).

### Task: Needle-in-a-haystack retrieval (multi-value)
- "(Multi-Value NIAH) We associate multiple values with one key and ask for all of them" (Section C.2. Evaluation)
- "We evaluate long-context retrieval using the needle-in-a-haystack (NIAH) setup." (Section C.2. Evaluation)
- Inference: Input/Output treated as token sequences (1D (t)) with capped context length based on "h_1, \ldots, h_T" and "fixed maximum context length" (Sections 2, C.2).

### Task: Long-context language modeling task (MultiFieldQA)
- "on four long context language modeling tasks from Bai et al. (2023)" (Table 2)
- "| Method            | MultiFieldQA | MuSiQue | GovReport | LCC   | NIAH  | Avg.  |" (Table 2)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are text responses for "long context language modeling tasks" (Section 2; Table 2).

### Task: Long-context language modeling task (MuSiQue)
- "on four long context language modeling tasks from Bai et al. (2023)" (Table 2)
- "| Method            | MultiFieldQA | MuSiQue | GovReport | LCC   | NIAH  | Avg.  |" (Table 2)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are text responses for "long context language modeling tasks" (Section 2; Table 2).

### Task: Long-context language modeling task (GovReport)
- "on four long context language modeling tasks from Bai et al. (2023)" (Table 2)
- "| Method            | MultiFieldQA | MuSiQue | GovReport | LCC   | NIAH  | Avg.  |" (Table 2)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are text responses for "long context language modeling tasks" (Section 2; Table 2).

### Task: Long-context language modeling task (LCC)
- "on four long context language modeling tasks from Bai et al. (2023)" (Table 2)
- "| Method            | MultiFieldQA | MuSiQue | GovReport | LCC   | NIAH  | Avg.  |" (Table 2)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are text responses for "long context language modeling tasks" (Section 2; Table 2).

### Task: Multiple-choice science QA (ARC-E)
- "ARC-E/C: grade-school science QA split into Easy and Challenge sets" (Section C.2. Evaluation)
- "six standard multiple-choice benchmarks" (Section C.2. Evaluation)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are discrete choice labels (0D, Fixed) based on "multiple-choice" (Sections 2, C.2).

### Task: Multiple-choice science QA (ARC-C)
- "ARC-E/C: grade-school science QA split into Easy and Challenge sets" (Section C.2. Evaluation)
- "six standard multiple-choice benchmarks" (Section C.2. Evaluation)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are discrete choice labels (0D, Fixed) based on "multiple-choice" (Sections 2, C.2).

### Task: Multiple-choice sentence completion (HellaSwag)
- "HellaSwag: adversarially filtered commonsense sentence completion" (Section C.2. Evaluation)
- "six standard multiple-choice benchmarks" (Section C.2. Evaluation)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are discrete choice labels (0D, Fixed) based on "multiple-choice" (Sections 2, C.2).

### Task: Multiple-choice open-book QA (OpenBookQA)
- "Open-BookQA: combining a small \"open book\" of science facts" (Section C.2. Evaluation)
- "six standard multiple-choice benchmarks" (Section C.2. Evaluation)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are discrete choice labels (0D, Fixed) based on "multiple-choice" (Sections 2, C.2).

### Task: Multiple-choice physical commonsense reasoning (PIQA)
- "PIQA: two-choice physical commonsense reasoning" (Section C.2. Evaluation)
- "six standard multiple-choice benchmarks" (Section C.2. Evaluation)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are discrete choice labels (0D, Fixed) based on "multiple-choice" (Sections 2, C.2).

### Task: Multiple-choice coreference/commonsense (WinoGrande)
- "WinoGrande: a large-scale, adversarial Winograd-style coreference/commonsense benchmark" (Section C.2. Evaluation)
- "six standard multiple-choice benchmarks" (Section C.2. Evaluation)
- Inference: Inputs are token sequences with capped context length based on "h_1, \ldots, h_T" and "C_{\text{train}} < C_{\text{test}}"; outputs are discrete choice labels (0D, Fixed) based on "multiple-choice" (Sections 2, C.2).
