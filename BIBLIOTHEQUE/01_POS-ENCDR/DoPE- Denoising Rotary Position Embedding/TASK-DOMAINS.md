# DoPE: Denoising Rotary Position Embedding (Not specified in the paper.)
Source: DoPE- Denoising Rotary Position Embedding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| information retrieval (needle-in-a-haystack) | long-context tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | synthesized relevant information | Not specified in the paper. | Not specified in the paper. |
| in-context learning (reasoning) | tokens (in-context exemplars + test example) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Not specified in the paper. | answers to MATH problems (inferred) | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates two text-only tasks: needle-in-a-haystack information retrieval and many-shot in-context learning for reasoning on MATH problems. Both tasks operate over long 1D token sequences with capped context lengths (e.g., 24K/64K for NIH and 16K for MICL), using standard causal self-attention; state dynamics are not specified. Output objects are described as synthesized information or inferred as problem answers, while output dimensionality and dynamics are not explicitly specified.

## Evidence
### Task: information retrieval (needle-in-a-haystack)
- "The \"needle-in-a-haystack\" synthesis task presents a particularly challenging problem in the field of natural language processing and information retrieval." (Section 5.1 Experimental Setup)
- "identify and synthesize highly relevant but sparse information from large volumes of data" (Section 5.1 Experimental Setup)
- Inference: Marked input as `1D (t)`, input dynamics as `Capped`, and attention as `Static` because the model uses "token representations  $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$", experiments use "context length—24K and 64K tokens", and attention is defined with "the causal mask". (Section 2.1 Multi-Head Self-Attention; Section 5.1 Experimental Setup)

### Task: in-context learning (reasoning)
- "tests whether the model can identify similar reasoning patterns from the context." (Section 5.3 Many-Shot In-Context Learning)
- "Experiments utilize In-Context Learning (ICL) constructed from the nlile/hendrycks-MATH-benchmark dataset." (Table 2 caption)
- Inference: Marked input as `1D (t)`, input dynamics as `Capped`, attention as `Static`, and output as problem answers based on "token representations  $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$", "L_{\\text{target}} = 16K for MICL experiments", "the causal mask", and "Results are accuracy scores on 100 sampled MATH problems." (Section 2.1 Multi-Head Self-Attention; Appendix A.2 Experimental Setup; Table 2 caption)
