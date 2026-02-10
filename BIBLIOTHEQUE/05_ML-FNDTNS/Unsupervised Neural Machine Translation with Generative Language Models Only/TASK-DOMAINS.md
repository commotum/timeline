# Unsupervised Neural Machine Translation with Generative Language Models Only (2021)
Source: Unsupervised Neural Machine Translation with Generative Language Models Only.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unsupervised neural machine translation (bidirectional generation) | source-language token sequences in translation prompts | 1D (t) | Not specified in the paper. | Static (inferred) | Direct (inferred) | target-language token sequences (translations) | 1D (t) | Not specified in the paper. |
| Autoregressive language modeling (text completion) | token prefix sequences (internet text or formatted monotext prompts) | 1D (t) | Not specified in the paper. | Static (inferred) | Direct (inferred) | token continuations (formatted bitext or natural-language continuation) | 1D (t) | Not specified in the paper. |

## Summary
The paper covers two text-sequence task intents: unsupervised neural machine translation and autoregressive language modeling used to implement that translation pipeline. For both tasks, the supported input and output objects are token sequences, so the justified dimensionality is 1D (t) throughout. The OCR text does not explicitly specify fixed maximum interface sizes, so input/output dynamics are not specified in the paper. From the described prompt-to-completion autoregressive setup, attention is Static and state is Direct (both inferred).

## Evidence
### Task: Unsupervised neural machine translation (bidirectional generation)
- "We target the domain of unsupervised neural machine translation (NMT), which typically involves bootstrapping a weak translation model before amplifying its translation ability via backtranslation." (Section 1 Introduction)
- "Given bitext  $\langle seq1, seq2 \rangle$  in languages  $L_1$  and  $L_2$ , we format the translation task as follows:" (Section 3 Backtranslation via Language Modeling)
- "At test-time, the LM is prompted with <code>[L1] <seq> [[TRANSLATE]] [L2]</code> and we parse a candidate translation <code><sampledSeq></code> from the sampled completion." (Section 3 Backtranslation via Language Modeling)
- Inference: Attention Dynamic is Static (inferred) and State Dynamic is Direct (inferred), because decoding is described as direct conditional completion from a fixed prompt ("$\tilde{\mathbf{x}} \sim p_{\boldsymbol{\theta}}(\cdot \mid f(\mathbf{y}))$") with no runtime retrieval policy or persistent constructed state described (Algorithm 1, Section 3).

### Task: Autoregressive language modeling (text completion)
- "In our present work, we cast machine translation as a language modeling task and jointly train and sample generations from a single language model for both source-to-target and target-to-source translation." (Section 3 Backtranslation via Language Modeling)
- "We assume that  $p_{\theta}(\cdot)$  has already been trained to complete formatted monotext ([L1] <seq1> [[TRANSLATE]] [L2]) to formatted bitext ([L1] <seq1> [[TRANSLATE]] [L2] <seq2>)." (Section 3 Backtranslation via Language Modeling)
- "1. Generatively pre-train a language model  $p_{\theta}(\cdot)$  on a large corpus of Internet data." (Section 4.1 Few-shot Amplification and Distillation)
- Inference: Attention Dynamic is Static (inferred) and State Dynamic is Direct (inferred), because the task is framed as autoregressive completion from provided prefixes without any described runtime mechanism for selecting external information or maintaining first-class constructed memory/state beyond token-context processing (Sections 3 and 4.1).
