# SELECTIVE ROTARY POSITION EMBEDDING (Not specified in the paper.)
Source: Selective Rotary Position Embedding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Retrieval (associative recall; MQAR) | token sequences with associative key-query structure (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | recalled associated values/tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Recall and memorization (MAD: Compress/Fuzzy Recall/In-Context Recall/Memorize/Noisy Recall/Selective Copy) | token sequences (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | task-specific recall/copied targets (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Sequence copying | token sequence plus <copy> token | 1D (t) (inferred) | Capped | Static (inferred) | Constructed (inferred) | copied token sequence | 1D (t) (inferred) | Capped (inferred) |
| State tracking (permutation composition/parity) | permutation-symbol sequence (inferred) | 1D (t) (inferred) | Capped | Static (inferred) | Constructed (inferred) | tracked permutation/parity state label (inferred) | 0D (inferred) | Fixed (inferred) |
| Generation (language modeling / next-token prediction) | text tokens | 1D (t) (inferred) | Capped | Static (inferred) | Constructed (inferred) | next-token predictions/generated tokens | 1D (t) (inferred) | Capped (inferred) |
| Prediction (cloze word prediction; LAMBADA) | text token context | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | predicted target word/token | 0D (inferred) | Fixed (inferred) |
| Classification (multiple-choice QA: PIQA/Hella./Wino./ARC-e/ARC-c) | text prompts with answer choices (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | answer option label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates Selective RoPE on synthetic sequence tasks and real-world language tasks, covering retrieval/recall, copying, state tracking, language modeling, and downstream QA-style evaluation. The supported inputs are token sequences (`1D (t)`), while outputs include both token sequences (`1D (t)`) and discrete labels (`0D`). Dynamics are explicitly capped where sequence limits are provided (e.g., context length 4096, copying max eval length 512, state-tracking train/eval lengths), while MQAR and MAD caps are not explicitly specified in the OCR text. Attention behavior is classified as `Static` and state as `Constructed` by inference from the causal prefix processing and explicit recurrent state equations.

## Evidence
### Task: Retrieval (associative recall; MQAR)
- "**MQAR.** We evaluate GLA + *Selective RoPE* on Multi-Query Associative Recall" (Section 4.2 Synthetic Language Tasks)
- "... improve performance in language modeling and on difficult sequence tasks like copying, state tracking, and retrieval." (Abstract)
- Inference: `1D (t)` input/output and `Static`/`Constructed` are inferred from the sequence+state formulation: "transforms a sequence of L inputs ... into the sequence of outputs" and "Here  $S_t ...$ are state ... recurrent form" (Section 2 Background).

### Task: Recall and memorization (MAD: Compress/Fuzzy Recall/In-Context Recall/Memorize/Noisy Recall/Selective Copy)
- "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024) which tests a model's ability to store and recall information within its context." (Section 4.2 Synthetic Language Tasks)
- "| Model               | Compress    | Fuzzy<br>Recall | In-Context<br>Recall | Memorize | Noisy<br>Recall | Selective<br>Copy | Average |" (Table 1, Section 4.2)
- Inference: Input/output dimensionality is marked `1D (t)` and attention/state as `Static`/`Constructed` from the same sequence and recurrent-state definitions in Section 2; explicit MAD interface caps are not stated in the OCR.

### Task: Sequence copying
- "This task differs from *Selective Copy* in MAD in that the entire input sequence has to be copied token-by-token after the model is presented with a <copy> token." (Section 4.2 Synthetic Language Tasks)
- "| Train task                      | copy      |" and "| Eval task                       | copy      |" (Table 6, Appendix B.2.4)
- Inference: `1D (t)` and `Out Dynamics = Capped` are inferred from token-by-token copying with bounded lengths, including "| Max length (eval)               | 512       |" (Table 6, Appendix B.2.4).

### Task: State tracking (permutation composition/parity)
- "**State Tracking.** A common way to evaluate the expressivity of a model is *state tracking* on permutation composition" (Section 4.2 Synthetic Language Tasks)
- "... parity ... amounts to permutation composition on the symmetric group of two elements,  $S_2$ ..." and "We also experiment on  $A_3$ ..." (Section 4.2 Synthetic Language Tasks)
- Inference: `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred as finite-state/parity label prediction; `In Dynamics = Capped` is supported by "Train sequence length 128 tokens" and "Eval sequence length 512 tokens" (Appendix B.2.2).

### Task: Generation (language modeling / next-token prediction)
- "Position information is essential for language modeling." (Abstract)
- "For our language modeling experiments we train ... All models are trained on 35B tokens ... at a context length of 4096 ..." (Section 4.3 Language Modeling)
- Inference: `Out Dimension = 1D (t)` and `Out Dynamics = Capped` are inferred from autoregressive token-sequence modeling under the stated context cap.

### Task: Prediction (cloze word prediction; LAMBADA)
- "| Model               | LMB.<br>ppl↓          | LMB.<br>acc ↑ | PIQA<br>acc ↑ | Hella.<br>acc_n ↑ | Wino.<br>acc ↑ | ARC-e<br>acc ↑ | ARC-c<br>acc_n ↑ | Avg.        |" (Table 2, Section 4.3)
- "For GLA, Selective RoPE reduces Lambada perplexity relative to RoPE ..." (Section 4.3 Language Modeling)
- Inference: This is treated as single-target token prediction (`0D`, `Fixed`) from token context; `Static`/`Constructed` follows Section 2's causal sequence processing and recurrent state definition.

### Task: Classification (multiple-choice QA: PIQA/Hella./Wino./ARC-e/ARC-c)
- "The best models are then evaluated on downstream tasks from lm-eval-harness (Gao et al., 2024) ..." (Section 4.3 Language Modeling)
- "We follow the default zero-shot evaluation setup in lm-eval-harness ... and report the macro-average accuracy over the core multiple-choice tasks ..." (Section 4.3 Language Modeling)
- Inference: Inputs are prompt-plus-choice token sequences (`1D (t)`), outputs are discrete option labels (`0D`, `Fixed`), and capped context is inferred from the same Section 4.3 model interface.
