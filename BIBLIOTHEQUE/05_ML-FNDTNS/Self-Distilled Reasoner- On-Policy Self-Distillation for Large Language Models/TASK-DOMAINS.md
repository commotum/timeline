# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models (Not specified in the paper)
Source: Self-Distilled Reasoner- On-Policy Self-Distillation for Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathematical reasoning generation | Problem statements (tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Solution/answer tokens | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper focuses on one task domain: generating solutions for competition-level mathematical reasoning problems using an autoregressive LLM. Both input and output are token sequences, which supports a 1D (t) characterization, and the interface is capped by explicit generation/sequence limits in the reported setup. The runtime setup describes fixed conditioning contexts for teacher/student prompts rather than runtime retrieval or observation selection, supporting Static attention. The method is framed as next-token distribution matching and generation, with no explicit external memory/search structure, supporting Direct state.

## Evidence
### Task: Mathematical reasoning generation
- "We evaluate OPSD on four competition-level mathematical reasoning tasks" (Section 1, Contributions)
- "We consider a dataset of problem-solution pairs  $\mathcal{S}=\{(x_i,y_i^\star)\}_{i=1}^N$ , where each  $x_i$  denotes a problem and  $y_i^\star$  is the corresponding reference solution" (Section 3.1)
- "Given a problem x, the student generates an on-policy response" (Section 3.2)
- "At each position n, they induce *next-token* distributions over  $y_n \in \mathcal{V}$  conditioned on the same student prefix" (Section 3.2)
- "each generation is capped at 2048 tokens for OPSD and 16384 tokens for GRPO" (Figure 3, Section 4.2)
- Inference: In Dimension and Out Dimension are labeled 1D (t) because the model operates over ordered token positions ("token-level" supervision and sequence indexing by position n in Section 3.2). In Dynamics and Out Dynamics are labeled Capped from explicit token caps (Figure 3) and fixed maximum training/evaluation lengths (Appendix 8.1, Tables 4-6). Attention Dynamic is labeled Static because the paper defines fixed conditioning contexts for teacher/student policies ("$p_T(\cdot \mid x, y^*)$" and "$p_S(\cdot \mid x)$" in Section 3.2) without runtime retrieval/selection. State Dynamic is labeled Direct because the method is described as autoregressive next-token prediction/distribution matching on current prompt and prefix (Sections 3.2 and Eq. 6), with no explicit constructed external state.
