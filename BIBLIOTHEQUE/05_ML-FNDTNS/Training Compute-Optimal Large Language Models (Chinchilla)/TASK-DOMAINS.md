# Training Compute-Optimal Large Language Models (2022)
Source: Training Compute-Optimal Large Language Models (Chinchilla).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token language modelling | Token sequences (previous tokens) | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token probabilities / generated tokens | 0D; 1D (t) | Capped (inferred) |
| Reading comprehension | Passage and question tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer option or final-word tokens | 0D; 1D (t) | Capped (inferred) |
| Closed-book question answering | Question tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer tokens | 1D (t) | Capped (inferred) |
| Common sense reasoning | Prompt tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer option tokens | 0D; 1D (t) | Capped (inferred) |
| MMLU exam question answering | Exam-like question tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer option tokens | 0D; 1D (t) | Capped (inferred) |
| BIG-bench multitask reasoning/completion | Task prompt tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Completion/answer tokens | 0D; 1D (t) | Capped (inferred) |
| Coreference resolution (Winogender) | Pronoun-and-occupation context tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Pronoun referent label | 0D | Fixed (inferred) |
| Unconditional text generation (toxicity analysis) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Direct (inferred) | Generated text samples | 1D (t) | Not specified in the paper. |

## Summary
The paper covers a text-only model across pretraining language modelling and multiple downstream NLP domains: reading comprehension, question answering, common sense benchmarks, MMLU, BIG-bench, plus bias/toxicity analyses. Inputs and outputs are predominantly token sequences, so the dominant address space is 1D (t), with some task outputs functioning as 0D labels in benchmark scoring. The only explicitly stated structural bound is a maximum sequence length (s_max), which supports Capped input/output dynamics for most text tasks. Attention and state are inferred as Static and Direct from the autoregressive transformer setup and next-token prediction framing.

## Evidence
### Task: Next-token language modelling
- "Formally, we consider the task of predicting the next token  $y \in \mathcal{Y}$  based on the previous tokens in a sequence  $x \in \mathcal{Y}^s$ , with s varying from 0 to  $s_{\text{max}}$ —the maximum sequence length." (Section D.2)
- "| Language Modelling    | 20      | WikiText-103, The Pile: PG-19, arXiv, FreeLaw,             |" (Table 5)
- Inference: In/Out Dynamics are marked Capped (inferred) because the OCR explicitly states a "maximum sequence length" (Section D.2). Attention Dynamic and State Dynamic are marked Static/Direct (inferred) because the model is described as an "Autoregressive Transformer Language Model" and the core formulation is next-token prediction from prior tokens (Section D.2; Table A8 model type).

### Task: Reading comprehension
- "| Reading Comprehension | 3       | RACE-m, RACE-h, LAMBADA                                    |" (Table 5)
- "On the final word prediction dataset LAMBADA (Paperno et al., 2016), *Chinchilla* achieves 77.4% accuracy, compared to 74.5% accuracy from *Gopher* and 76.6% from MT-NLG 530B (see Table 7). On RACE-h and RACE-m (Lai et al., 2017), *Chinchilla* greatly outperforms *Gopher*, improving accuracy by more than 10% in both cases—see Table 7." (Section 4.2.3)
- Inference: The row uses 1D (t) text inputs and Capped/Static/Direct (inferred) by reusing the paper’s autoregressive next-token architecture and maximum sequence length framing (Section D.2, Section 4.1).

### Task: Closed-book question answering
- "| Question Answering    | 3       | Natural Questions, TriviaQA, TruthfulQA                    |" (Table 5)
- "Results on closed-book question answering benchmarks are reported in Table 9." (Section 4.2.6)
- Inference: Output is represented as answer tokens with 1D (t), and dynamics/attention/state are marked Capped/Static/Direct (inferred) from the same autoregressive token-sequence setup with maximum sequence length (Section D.2).

### Task: Common sense reasoning
- "| Common Sense          | 5       | HellaSwag, Winogrande, PIQA, SIQA, BoolQ                   |" (Table 5)
- "We evaluate *Chinchilla* on various common sense benchmarks: PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), Winogrande (Sakaguchi et al., 2020), HellaSwag (Zellers et al., 2019), and BoolQ (Clark et al., 2019)." (Section 4.2.5)
- Inference: Input/output are treated as text prompts plus selected/generated answers; 0D is included for choice-style outputs. Capped/Static/Direct are inferred from the transformer next-token formulation and sequence-length cap (Section D.2).

### Task: MMLU exam question answering
- "| MMLU                  | 57      | High School Chemistry, Astronomy, Clinical Knowledge,      |" (Table 5)
- "The Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020) consists of a range of exam-like questions on academic subjects." (Section 4.2.2)
- Inference: The row maps MMLU to text question-answering over token sequences; 0D is included for choice outputs. Capped/Static/Direct are inferred from the same autoregressive architecture and sequence cap (Section D.2, Section 4.1).

### Task: BIG-bench multitask reasoning/completion
- "| BIG-bench             | 62      | Causal Judgement, Epistemic Reasoning, Temporal Sequences, |" (Table 5)
- "We analysed *Chinchilla* on the same set of BIG-bench tasks (BIG-bench collaboration, 2021) reported in Rae et al. (2021)." (Section 4.2.4)
- Inference: BIG-bench is represented as text prompt-to-answer/completion tasks in 1D (t), with some outputs treated as 0D choice labels. Capped/Static/Direct are inferred from the autoregressive transformer and maximum sequence length statement (Section D.2).

### Task: Coreference resolution (Winogender)
- "Here, we test if potential gender and occupation biases manifest in unfair outcomes on coreference resolutions, using the Winogender dataset (Rudinger et al., 2018) in a zero-shot setting." (Section 4.2.7)
- "Winogender tests whether a model can correctly determine if a pronoun refers to different occupation words." (Section 4.2.7)
- Inference: Output is labeled 0D with Fixed (inferred) because the task is framed as deciding pronoun referent outcomes per instance. Capped/Static/Direct are inferred from the same autoregressive token-processing setup (Section D.2).

### Task: Unconditional text generation (toxicity analysis)
- "Sample toxicity. Language models are capable of generating toxic language—including insults, hate speech, profanities and threats (Gehman et al., 2020; Rae et al., 2021)." (Section 4.2.7)
- "Similar to the protocol of Rae et al. (2021), we generate 25,000 unprompted samples from *Chinchilla*, and compare their *PerspectiveAPI* toxicity score distribution to that of *Gopher*-generated samples." (Section 4.2.7)
- Inference: Attention Dynamic and State Dynamic are marked Static/Direct (inferred) from the shared autoregressive transformer setup. Input and dynamics are left as "Not specified in the paper." because the OCR does not specify a concrete conditioning input or generation-length interface for this evaluation.
