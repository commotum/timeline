# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Not specified in the paper.)
Source: Megatron-LM- Training Multi-Billion Parameter Language Models Using Model Parallelism.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language modeling (left-to-right generation) | subword token sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | next-token prediction / token probabilities (inferred) | 1D (t) (inferred) | Capped (inferred) |
| masked language modeling | token sequence with masks (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | masked token prediction (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| sentence order prediction | sentence pair tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | order label (inferred) | 0D (inferred) | Not specified in the paper. |
| cloze-style reading comprehension / word prediction (LAMBADA) | context word tokens with one token masked | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | missing token (word/subword tokens) | 1D (t) (inferred) | Not specified in the paper. |
| classification (MNLI) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label (inferred) | 0D (inferred) | Not specified in the paper. |
| classification (QQP) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label (inferred) | 0D (inferred) | Not specified in the paper. |
| question answering (SQuAD 1.1/2.0) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| reading comprehension (RACE) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper covers text-based NLP tasks centered on language modeling (left-to-right GPT-2 and masked LM for BERT), plus sentence order prediction, cloze-style word prediction, and downstream evaluation on MNLI, QQP, SQuAD, and RACE. Where described, inputs are token sequences (1D (t)), with GPT-2 trained on fixed 1024-token windows, implying capped dynamics and static attention for that task. For several downstream tasks, the paper names datasets without specifying task I/O, and state dynamics are not described.

## Evidence
### Task: language modeling (left-to-right generation)
- "we focus on GPT-2 (Radford et al., 2019), a left-to-right generative transformer based language model" (Section 4. Setup)
- "all training is performed with sequences of 1024 subword units" (Section 4.2)
- "transformers operate on a fixed window input size." (Section E.1)
- "language models which represent a probability distribution over entire sentences or texts." (Section E.1)
- Inference: Assigned 1D (t) dimensions plus capped dynamics and static attention from the fixed 1024-token window, and treated outputs as next-token probabilities based on the language-model probability distribution description above.

### Task: masked language modeling
- "BERT (Devlin et al., 2018), a bi-directional transformer model based on language model masking." (Section 4. Setup)
- "use whole word n-gram masking" (Section 4.2)
- Inference: Interpreted masking as token-level inputs with masked-token prediction and 1D (t) sequencing.

### Task: sentence order prediction
- "replace the next sentence prediction head with sentence order prediction" (Section 4.2)
- Inference: Treated the task as predicting a sentence-order label from sentence-pair text, yielding a 0D output.

### Task: cloze-style reading comprehension / word prediction (LAMBADA)
- "Clozestyle reading comprehension uses a context of word tokens  $x = x_{1:t}$  with one token  $x_j$  masked;" (Section E.2)
- "the models objective is to correctly predict the value of the missing  $j^{th}$  token." (Section E.2)
- "require that our model predict the multiple subword tokens that make up the word token." (Section E.2)
- Inference: Assigned 1D (t) dimensions because inputs and outputs are word/subword token sequences.

### Task: classification (MNLI) (inferred)
- "MNLI and QQP from the GLUE benchmark (Wang et al., 2019)" (Section 5.3)
- "accuracy    | accuracy" (Table 5, Section 5.3)
- Inference: Labeled MNLI as a classification task with label outputs because results are reported as accuracy.

### Task: classification (QQP) (inferred)
- "MNLI and QQP from the GLUE benchmark (Wang et al., 2019)" (Section 5.3)
- "accuracy    | accuracy" (Table 5, Section 5.3)
- Inference: Labeled QQP as a classification task with label outputs because results are reported as accuracy.

### Task: question answering (SQuAD 1.1/2.0)
- "SQuAD 1.1 and SQuAD 2.0 from the Stanford Question answering dataset" (Section 5.3)

### Task: reading comprehension (RACE)
- "the reading comprehension RACE dataset (Lai et al., 2017)" (Section 5.3)
