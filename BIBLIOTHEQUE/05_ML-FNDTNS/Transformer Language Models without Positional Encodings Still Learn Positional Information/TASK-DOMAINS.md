# Transformer Language Models without Positional Encodings Still Learn Positional Information (2022)
Source: Transformer Language Models without Positional Encodings Still Learn Positional Information.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction (causal language modeling) | Text token sequences | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next tokens / token probabilities | 1D (t) (inferred) | Capped (inferred) |
| Masked-token prediction (masked language modeling) | Masked text token sequences | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Predicted masked tokens | 1D (t) (inferred) | Capped (inferred) |
| Absolute-position classification from token representations (probe) | Token hidden representations across layers | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Absolute position class (0-1023) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers text-sequence tasks in the 1D (t) domain, centered on language modeling with and without explicit positional encodings. It evaluates both autoregressive next-token prediction and masked-token prediction, and adds an auxiliary probe task that classifies absolute token position from hidden representations. Across these tasks, sequence handling is bounded by explicit token limits (e.g., 128, 512, 1024, 2048), so the supported dynamics are best justified as Capped (inferred). Attention and state are not labeled in the paper, but the described setups support Static attention and Direct state assignments (both inferred).

## Evidence
### Task: Next-token prediction (causal language modeling)
- "Intuitively, encoding positional information explicitly is crucial for enabling transformer language models to predict the next token in a sequence." (Section 3 Experiment Setup)
- "The next token prediction is conditioned on the previous tokens in the sequence, and so we shuffle the order of the tokens in the prefix and compute the loss only for that specific token." (Appendix B Word Order Analysis)
- Inference: Input/Output were mapped to text token sequences and next-token outputs from the quoted language-model objective text. 1D (t), Capped, Static, and Direct are inferred from sequence processing with explicit sequence-length limits and no runtime retrieval/controller mechanism described (Sections 3, 6, and Appendix B).

### Task: Masked-token prediction (masked language modeling)
- "To test our hypothesis, we run similar experiments for masked language models (MLM) (Devlin et al., 2019), which use order-invariant attention (since no causal mask is applied)." (Section 1 Introduction)
- "We tested this corollary by training a masked language model based on RoBERTa large (Liu et al., 2019) on the Pile (see App. C for hyperparameters)." (Section 6 Conjecture)
- Inference: The MLM row maps masked-token reconstruction to 1D (t) token sequences with Capped dynamics based on explicit per-sequence token limits ("processes 128 tokens per sequence" in Table 4 caption). Static and Direct are inferred because the paper describes fixed-sequence processing without runtime selection or external persistent state construction for this task.

### Task: Absolute-position classification from token representations (probe)
- "Specifically, we train classifiers to predict the position of a token given its representation across different layers in the network." (Section 1 Introduction)
- "Specifically, we use the tokens' last hidden representation after each transformer layer, produced by the evaluated LM, and train a 2-layer feed-forward ReLU network to predict the absolute position (0 to 1023) of each token (i.e., as a multiclass classification problem)." (Section 5 Analysis)
- Inference: The task intent is classification, with output mapped to 0D class labels because the probe predicts one absolute-position class per token. 1D (t) input structure and Capped input dynamics are inferred from token-indexed sequence representations and the explicit 0-1023 position range; Static and Direct are inferred from the feed-forward probe setup described in Section 5.
