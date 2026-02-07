# KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation (Year not specified in the paper)
Source: KERPLE- Kernelized Relative Positional Embedding for Length Extrapolation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Causal language modeling | Tokens (OpenWebText2) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token predictions (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Causal language modeling | Tokens (GitHub) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token predictions (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Causal language modeling | Tokens (ArXiv) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token predictions (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Causal language modeling | Tokens (Wikitext-103) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token predictions (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates causal language modeling across multiple text domains: OpenWebText2 (internet text), GitHub (code), ArXiv (LaTeX academic papers), plus an additional Wikitext-103 experiment. Inputs are token sequences with lengths trained at 512 and evaluated up to 16384, indicating 1D (t) capped sequences. Attention is standard self-attention with causal masking (static, inferred), and outputs are next-token predictions with direct state (inferred).

## Evidence
### Task: Causal language modeling (OpenWebText2)
- "Note we focus on causal language modeling following ALiBi, so the matrices are triangular." (Figure 1 caption)
- "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description)
- "OpenWebText2 includes recent content from Reddit submissions until 2020" (Section 5.1 Dataset and Implementation Description)
- Inference: In/Out Dimension and Dynamics labeled `1D (t)`/`Capped` based on "Let  $\{w_m\}_{m=1}^L$  be the input tokens to a transformer model, where L is the total number of tokens." (Section 2.1 Preliminary) and "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results). Attention labeled `Static` based on "the self-attention module computes the scaled attention scores and generates the output vector  $\boldsymbol{o}_m$  at position m as:" (Section 2.1 Preliminary). Output `Next-token predictions` and State `Direct` inferred from the causal language modeling framing.

### Task: Causal language modeling (GitHub)
- "Note we focus on causal language modeling following ALiBi, so the matrices are triangular." (Figure 1 caption)
- "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description)
- "GitHub includes open-source repositories written in primary coding languages such as Java, C/C++, Python, and Go." (Section 5.1 Dataset and Implementation Description)
- Inference: In/Out Dimension and Dynamics labeled `1D (t)`/`Capped` based on "Let  $\{w_m\}_{m=1}^L$  be the input tokens to a transformer model, where L is the total number of tokens." (Section 2.1 Preliminary) and "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results). Attention labeled `Static` based on "the self-attention module computes the scaled attention scores and generates the output vector  $\boldsymbol{o}_m$  at position m as:" (Section 2.1 Preliminary). Output `Next-token predictions` and State `Direct` inferred from the causal language modeling framing.

### Task: Causal language modeling (ArXiv)
- "Note we focus on causal language modeling following ALiBi, so the matrices are triangular." (Figure 1 caption)
- "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description)
- "ArXiv includes papers written in LaTex in Math, Computer Science, Physics, and some related fields." (Section 5.1 Dataset and Implementation Description)
- Inference: In/Out Dimension and Dynamics labeled `1D (t)`/`Capped` based on "Let  $\{w_m\}_{m=1}^L$  be the input tokens to a transformer model, where L is the total number of tokens." (Section 2.1 Preliminary) and "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results). Attention labeled `Static` based on "the self-attention module computes the scaled attention scores and generates the output vector  $\boldsymbol{o}_m$  at position m as:" (Section 2.1 Preliminary). Output `Next-token predictions` and State `Direct` inferred from the causal language modeling framing.

### Task: Causal language modeling (Wikitext-103)
- "Note we focus on causal language modeling following ALiBi, so the matrices are triangular." (Figure 1 caption)
- "In this subsection, we present additional experiments on (a) large models, (b) longer training length, and (c) Wikitext-103." (Section A.4 Experiments on Large Model, Longer Training Length, and Wikitext-103)
- "Perplexity Comparison on Wikitext-103." (Table 10 caption)
- Inference: In/Out Dimension and Dynamics labeled `1D (t)`/`Capped` based on "Let  $\{w_m\}_{m=1}^L$  be the input tokens to a transformer model, where L is the total number of tokens." (Section 2.1 Preliminary) and "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results). Attention labeled `Static` based on "the self-attention module computes the scaled attention scores and generates the output vector  $\boldsymbol{o}_m$  at position m as:" (Section 2.1 Preliminary). Output `Next-token predictions` and State `Direct` inferred from the causal language modeling framing and the perplexity-based evaluation.
