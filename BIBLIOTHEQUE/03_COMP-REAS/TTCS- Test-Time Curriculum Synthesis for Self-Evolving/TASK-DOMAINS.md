# TTCS: Test-Time Curriculum Synthesis for Self-Evolving (Year not specified in the paper)
Source: TTCS- Test-Time Curriculum Synthesis for Self-Evolving.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Curriculum question generation | Test questions / reference questions (text) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Synthetic question variants (text) | 1D (t) (inferred) | Capped (inferred) |
| Reasoning question answering / problem solving | Mathematical and general-domain reasoning questions (test + synthetic) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Reasoning responses, solutions, predicted options/key phrases | 1D (t) (inferred); 0D (inferred) | Capped (inferred) |
| Synthetic-question quality assessment (difficulty/reward scoring) | Synthetic questions and sampled solver responses | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Self-consistency score and composite reward | 0D (inferred) | Fixed (inferred) |

## Summary
TTCS covers three text-centric tasks: generating curriculum questions, solving reasoning questions, and scoring synthetic-question quality for co-evolution. The primary I/O modality is token sequences (1D (t)), with scalar (0D) outputs for assessment scores/rewards. Dynamics are mostly Capped for question/response generation, while score outputs are Fixed as single scalars. The framework behavior is best characterized as Dynamic attention and Constructed state because runtime feedback (majority-vote consistency and filtering) is fed back into iterative policy updates.

## Evidence
### Task: Curriculum question generation
- "TTCS initializes two policies from the same pretrained model: a question synthesizer and a reasoning solver." (Abstract)
- "At the t-th iteration, the synthesizer generates M auxiliary questions  $\{x_i'\}_{i=1}^M$  for each RL rollout group conditioned on  $x_{\text{test}}$  with a well-designed prompt template (see Appendix D) as follows:" (Section 4.1)
- Inference: 1D (t), Capped, Dynamic, and Constructed are inferred because the synthesizer maps question text to question text, uses finite rollout groups ("generates M auxiliary questions"), and is updated in a feedback loop where "the solver's current performance provides a capability-aware training signal that shapes the synthesizer's generation distribution" (Section 4).

### Task: Reasoning question answering / problem solving
- "the solver performs online self-evolving on a mixture of synthetic questions and test questions, guided by self-consistency rewards" (Section 4)
- "given a training question  $x \in \mathcal{B}^t_{\text{train}}$ , the solver  $\pi^t_{\theta}$  first generates multiple reasoning responses via repeated high temperature sampling" (Section 4.2)
- "For the general-domain benchmarks BBEH, MMLU-Pro, and SuperG-PQA, we report Exact Match accuracy via greedy decoding (temperature T=0.0). We extract the predicted option or key phrase and check for a strict match against the ground truth label." (Section A.2)
- Inference: 1D (t) input/output and 0D final-answer behavior are inferred from question/response generation and option/key-phrase extraction; Capped is inferred from finite rollout/decoding setups (Sections 4.2, A.2); Dynamic and Constructed are inferred from iterative online filtering and policy updates ("retains only samples satisfying  $|s(x) - 0.5| \le \delta$ " in Section 4.2).

### Task: Synthetic-question quality assessment (difficulty/reward scoring)
- "we employ the solver as an online assessor to assign a composite reward to each synthetic question in rollout stage." (Section 4.1)
- "we define the self-consistency score to measure the difficulty of  $x_i'$  as:  $$s(x_i') = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}[y_{i,k} = \hat{y}_i].$$" (Section 4.1)
- "By combining the capability objective with diversity constraints, we define the final reward as follows to guide the training process:" (Section 4.1)
- Inference: 0D and Fixed output are inferred because each question receives scalar score/reward values; Dynamic and Constructed are inferred because the reward is computed from runtime sampled responses and then used to update the synthesizer (Section 4.1).
