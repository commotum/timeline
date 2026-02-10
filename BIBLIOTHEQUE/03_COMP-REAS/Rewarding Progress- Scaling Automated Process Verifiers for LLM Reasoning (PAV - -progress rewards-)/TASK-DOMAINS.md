# Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning (2024)
Source: Rewarding Progress- Scaling Automated Process Verifiers for LLM Reasoning (PAV - -progress rewards-).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathematical reasoning generation | Math problem text and reasoning-prefix tokens | 1D (t) | Capped | Static (inferred) | Direct | Multi-step reasoning trace with final answer tokens | 1D (t) | Capped |
| Outcome verification (ORM correctness prediction) | Problem-response pair `(x, y)` as token sequence | 1D (t) | Capped | Static (inferred) | Direct (inferred) | Final-answer correctness score `Rex(y, y_x^*)` | 0D | Fixed |
| Process progress verification (PRM/PAV step scoring) | State prefix and next-step tokens `(s_h, a_h)` | 1D (t) | Capped | Static (inferred) | Direct | Step-level score `Q^\pi(s_h, a_h)` or `A^\mu(s_h, a_h)` | 0D | Fixed |
| Planted sub-sequence generation (didactic) | Token-sequence prefix states over a fixed-length sequence | 1D (t) | Fixed | Static (inferred) | Direct | Token sequence that contains planted sub-sequence `y^*` | 1D (t) | Fixed |

## Summary
The paper primarily covers text-domain reasoning over token sequences, centered on solving math problems with multi-step responses and verifying those responses. Across the main models, inputs and outputs are 1D (t) sequences with capped rollout length, while verifier outputs are 0D scalar correctness/progress scores. The didactic analysis adds a second generation task with fixed-length token sequences. Attention behavior is not explicitly labeled, but the described rollout/scoring setup supports static attention as an inference.

## Evidence
### Task: Mathematical reasoning generation
- "Given a math problem  $x \in X$ , our goal is to improve a *base policy*  $\pi$  that samples a response  $y \sim \pi(\cdot \mid x)$  in the set  $\mathcal{Y}$ ." (Section 2. Preliminaries, Definitions, and Notation)
- "A response y consists of multiple reasoning steps (maximum H), separated by a delimiter ('next line' in our case), *i.e.*,  $y = (a_1, a_2, \ldots, a_H)$ ." (Section 2. Preliminaries, Definitions, and Notation)
- Inference: `Attention Dynamic = Static (inferred)` from autoregressive rollout over current prefixes/states ("we can view each step as an action" and beam expansion over sampled next steps), with no runtime retrieval/observation-selection mechanism described. (Section 2; "Using PRMs for beam search at test-time")

### Task: Outcome verification (ORM correctness prediction)
- "Given a response y, an ORM estimates the ground-truth correctness  $\text{Rex}(y,y_x^\star)$ ." (Section 2. Preliminaries, Definitions, and Notation)
- "Then we train an ORM that takes as input a problem-response pair (x,y) and predicts  $\text{Rex}(y,y_x^\star)$ ." (Section 2. Preliminaries, Definitions, and Notation)
- Inference: `State Dynamic = Direct (inferred)` and `Attention Dynamic = Static (inferred)` because ORM is defined as a direct mapping from `(x, y)` to a scalar correctness prediction, with no constructed external state or dynamic information-selection mechanism specified. (Section 2)

### Task: Process progress verification (PRM/PAV step scoring)
- "An outcome reward model (ORM) is a trained verifier that assigns a numerical score after the last step of the trace, and a process reward model (PRM) is a trained verifier that scores each step of the trace individually." (Section 2. Preliminaries, Definitions, and Notation)
- "For either of these, we need access to verifiers that are trained to predict the advantage  $A^{\mu}(s_h, a_h)$  under the prover. We refer to these verifiers as **process advantage verifiers** (**PAVs**)." (Section 3.2. Our Approach: Process Advantage Verifiers (PAV))
- Inference: `Attention Dynamic = Static (inferred)` because scoring is defined per provided `(state prefix, action)` sample (Eq. 1 / Eq. 5 context) rather than via runtime-chosen external context. (Section 2; Section 3.2)

### Task: Planted sub-sequence generation (didactic)
- "Given an unknown sub-sequence  $y^*$  consisting of tokens from vocabulary  $\mathcal{V} := \{1, 2, \dots, 15\}$ , we train a policy  $\pi$  to produce a response which contains this sub-sequence." (Section 3.3. Analysis in a Didactic Setting: Learning a Planted Sub-sequence)
- "We consider sequences of length 10 from a 15-token vocabulary  $\mathcal{V} := \{1, 2, \dots, 14\}$ ..." (Appendix B. Didactic Analysis)
- Inference: `Attention Dynamic = Static (inferred)` from fixed-length token rollouts in the didactic setup with no dynamic external retrieval described. (Section 3.3; Appendix B)
