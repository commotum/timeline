# Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (2024)
Source: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathematical problem solving (answer generation) | Natural-language math question prompts (tokens) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Natural-language step-by-step solutions and final answers (tokens) | 1D (t) (inferred) | Capped (inferred) |
| Step-level correctness verification (scoring) | Step-by-step candidate solutions and intermediate steps (tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Per-step correctness/value scores (0-1) and final aggregated answer score | 1D (t); 0D (inferred) | Capped (inferred) |

## Summary
This paper covers math reasoning on text prompts and an auxiliary verification task that scores solution steps. Inputs and generated solutions are token sequences, so the justified dimensionality is primarily 1D (t), with verifier outputs spanning per-step sequences and scalar aggregates. Dynamics are capped in practice because evaluation and search use explicit generation budgets and bounded procedures. The workflow includes dynamic behavior for adaptive test-time computation and constructed state during revision/search, while verifier scoring itself is a more direct, static mapping from provided steps to scores.

## Evidence
### Task: Mathematical problem solving (answer generation)
- "we focus on the MATH [13] benchmark, which consists of high-school competition level math problems with a range of difficulty levels." (Section 4, Datasets)
- "Formally, define  $Target(\theta, N, q)$  as the distribution over natural language output tokens induced by the model for a given prompt q, using test-time compute hyper-parameters  $\theta$ , and a compute budget of N." (Section 3.1)
- "Given a finetuned revision model, we can then sample a sequence of revisions from the model at test-time." (Section 6.1)
- Inference: 1D (t) input/output and capped dynamics are inferred from token-based prompt/response framing plus bounded generation procedures described in the paper (e.g., generation budgets and limited revision/search procedures). Dynamic attention and constructed state are inferred from adaptive, prompt-dependent compute allocation and sequential revision/search over prior attempts.

### Task: Step-level correctness verification (scoring)
- "a process reward model (PRM), which produces a prediction of the correctness of each intermediate step in an solution, rather than just the final answer." (Section 2)
- "We finetune our PRM as a binary classifier, where the model predicts a value between 0 and 1 at each step in the solution." (Appendix D)
- "We run this algorithm until the end of a solution or the maximum number of rounds of beam expansion are attained (40 in our case)." (Section 5.2)
- Inference: Input/output dimensions are inferred as token-sequence steps (1D (t)) with both per-step outputs and final aggregated scalar scores (0D). Attention and state are labeled Static/Direct (inferred) because the verifier is described as scoring provided steps, without explicit retrieval behavior or explicit constructed memory specific to the verification mapping.
