# Scaling Embedding Layers in Language Models (2025)
Source: Scaling Embedding Layers in Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-word prediction (causal language modeling) | Sequence of tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Probability distribution over next token | 0D (inferred) | Fixed (inferred) |
| Downstream benchmark evaluation (zero-shot/post-training) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper’s explicit primary task is causal language modeling via next-word prediction over token sequences. For this task, the OCR supports a capped 1D token input and a single next-token distribution output; attention/state classifications are inferred from the fixed-window decoder setup. The paper also evaluates downstream benchmark performance (zero-shot and post-training), but the OCR does not explicitly specify those benchmarks’ task I/O structures in glossary terms.

## Evidence
### Task: Next-word prediction (causal language modeling)
- "We focus on pre-training decoder-only language models with causal language modeling [Radford et al., 2019]." (Section 2 Preliminaries)
- "Output: Probability distribution over next token \hat{\sigma}_{m+1}." (Algorithm 2)
- "Input: (\sigma_1, \dots, \sigma_m) \in V_{\mathrm{token}}^* for m \leq T, where T is the maximum sequence length." (Algorithm 4)
- Inference: `1D (t)` is inferred from token *sequence* input; `Capped` is inferred from "m \leq T" and "maximum sequence length"; `0D` and `Fixed` are inferred because the model outputs one next-token distribution per invocation; `Static` and `Direct` are inferred from the fixed-sequence decoder mapping without explicit runtime retrieval/control loops (Algorithms 2 and 4).

### Task: Downstream benchmark evaluation (zero-shot/post-training)
- "We report zero-shot accuracy on six standard downstream benchmarks: MMLUvar, Hellaswag, ARC-Challenge, ARC-Easy, CommonsenseQA (CSQA), and PIQA." (Section 4.2)
- "We apply SCONE to supervised fine-tuning of Qwen3-4B-base ... Table 4 compares the resulting SCONE-enabled models with the Qwen3-4B baseline in terms of both accuracy and decoding latency." (Section E.3)
