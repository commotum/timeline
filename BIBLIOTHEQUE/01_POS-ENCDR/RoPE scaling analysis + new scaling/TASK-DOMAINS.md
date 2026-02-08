# Extending Context Window of Large Language Models from a Distributional Perspective (Not specified in the paper.)
Source: RoPE scaling analysis + new scaling.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling / text generation | tokens | 1D (t) | Capped | Static (inferred) | Direct (inferred) | tokens | 1D (t) | Capped (inferred) |
| LongBench-E benchmark tasks (individual task intents not specified in the paper.) | tokens | 1D (t) | Capped | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| TruthfulQA (task intent not specified in the paper.) | tokens (inferred) | 1D (t) (inferred) | Capped | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Hellaswag (task intent not specified in the paper.) | tokens (inferred) | 1D (t) (inferred) | Capped | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| MMLU (task intent not specified in the paper.) | tokens (inferred) | 1D (t) (inferred) | Capped | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| ARC-c (task intent not specified in the paper.) | tokens (inferred) | 1D (t) (inferred) | Capped | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Passkey retrieval | tokens | 1D (t) | Capped | Static (inferred) | Direct (inferred) | tokens (pass key) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Long-context retrieval (RULER benchmark) | tokens | 1D (t) | Capped | Static (inferred) | Direct (inferred) | tokens (retrieved information) (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates a text-only LLM setup across language modeling/text generation and retrieval-oriented benchmark settings, including LongBench-E, short-context benchmark tasks, passkey retrieval, and RULER. The directly supported data modality is token sequences, which maps to 1D (t) task structure in the glossary. Input dynamics are consistently Capped because the paper explicitly evaluates bounded context windows (e.g., 4k, 8k, 16k, and 20k). Attention/State behavior is not labeled directly in the paper, but the RoPE-based autoregressive formulation supports Static attention and Direct state as inferences.

## Evidence
### Task: Language modeling / text generation
- "LLMs generate language sequences by sampling from the learned distribution  $p(x) = \prod_m p(x_m|x_{< m})$" (Section 3.1 Rotary Angle Distribution)
- "Perplexity is commonly employed to evaluate a model's language modeling capabilities" (Section B.2.3 Perplexity)
- Inference: Attention Dynamic = Static and State Dynamic = Direct are inferred from the autoregressive RoPE attention setup, where the model conditions on the provided sequence rather than runtime retrieval/control; this is supported by "Suppose the input of a single attention head is  $x_1, \dots, x_l \in \mathbb{R}^d$ , where l is the sequence length" (Section 2.1 Rotary Position Embedding (RoPE)). Out Dynamics = Capped is inferred from explicit bounded context-window settings (Sections 4.1 and 4.4).

### Task: LongBench-E benchmark tasks (individual task intents not specified in the paper.)
- "We utilize the Longbench-E benchmark (Bai et al., 2023), which is specifically designed for evaluating models with long context window." (Section 4.2 Long Context Evaluation)
- "The Longbench-E benchmark consists of 13 diverse tasks, with the average length of most tasks ranging from 5k to 15k." (Section 4.2 Long Context Evaluation)
- Inference: Input/Output are marked as tokens and 1D (t) based on the paper's LLM sequence formulation (Section 3.1) and sequence-length framing of the benchmark. Individual task intents are not enumerated in the OCR text, so the task intent detail is marked as not specified.

### Task: TruthfulQA (task intent not specified in the paper.)
- "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." (Section 4.3 Short Context Validation)
- "Table 2: Comparative performance of various context window extension methods relative to the original LLaMA2 on the Hugging Face Open LLM benchmark." (Section 4.3 Short Context Validation)
- Inference: Input/Output and dimensions are inferred as token-sequence processing (1D (t)) under the same autoregressive LLM setup; Attention = Static and State = Direct are inferred for the same reason as above. Exact task intent beyond benchmark naming is not explicitly defined in the OCR text.

### Task: Hellaswag (task intent not specified in the paper.)
- "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." (Section 4.3 Short Context Validation)
- "Table 2: Comparative performance of various context window extension methods relative to the original LLaMA2 on the Hugging Face Open LLM benchmark." (Section 4.3 Short Context Validation)
- Inference: Input/Output and dimensions are inferred as token-sequence processing (1D (t)) under the same autoregressive LLM setup; Attention = Static and State = Direct are inferred for the same reason as above. Exact task intent beyond benchmark naming is not explicitly defined in the OCR text.

### Task: MMLU (task intent not specified in the paper.)
- "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." (Section 4.3 Short Context Validation)
- "Table 2: Comparative performance of various context window extension methods relative to the original LLaMA2 on the Hugging Face Open LLM benchmark." (Section 4.3 Short Context Validation)
- Inference: Input/Output and dimensions are inferred as token-sequence processing (1D (t)) under the same autoregressive LLM setup; Attention = Static and State = Direct are inferred for the same reason as above. Exact task intent beyond benchmark naming is not explicitly defined in the OCR text.

### Task: ARC-c (task intent not specified in the paper.)
- "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." (Section 4.3 Short Context Validation)
- "Table 2: Comparative performance of various context window extension methods relative to the original LLaMA2 on the Hugging Face Open LLM benchmark." (Section 4.3 Short Context Validation)
- Inference: Input/Output and dimensions are inferred as token-sequence processing (1D (t)) under the same autoregressive LLM setup; Attention = Static and State = Direct are inferred for the same reason as above. Exact task intent beyond benchmark naming is not explicitly defined in the OCR text.

### Task: Passkey retrieval
- "We further evaluate the model's ability to retrieve a simple passkey from a massive amount of text via passkey retrieval task (Mohtashami and Jaggi, 2023)." (Section 4.4 Passkey Retrieval)
- "What is the pass key? The pass key is" (Section B.3 Passkey Prompt)
- Inference: Output is represented as tokens (the recovered passkey string), with 1D (t) output dimension. Attention = Static and State = Direct are inferred from the same fixed-context autoregressive architecture. Out Dynamics = Capped is inferred from bounded evaluation settings ("we set the maximum input length for all models to 20k", Section 4.4).

### Task: Long-context retrieval (RULER benchmark)
- "The RULER (Hsieh et al., 2024) benchmark is employed to evaluate the long-context retrieval capabilities of models" (Section B.2.1 RULER Benchmark)
- "all methods have enhanced the model's ability to retrieve information from long documents, with our approach achieving the highest retrieval accuracy." (Section B.2.1 RULER Benchmark)
- Inference: Output as tokens and 1D (t) output dimension is inferred from the shared LLM sequence-generation setup; Attention = Static and State = Direct are inferred from the same architecture and fixed context-window interface.
