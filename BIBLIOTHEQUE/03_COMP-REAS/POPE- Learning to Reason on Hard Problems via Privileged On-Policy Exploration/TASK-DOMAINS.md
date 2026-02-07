# POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration (Not specified in the paper.)
Source: POPE- Learning to Reason on Hard Problems via Privileged On-Policy Exploration.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathematical problem solving (reasoning) | math problem prompts (text) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | text solutions/answers (rollouts) | 1D (t) (inferred) | Capped (inferred) |
| Coding problem solving (reasoning) | coding problem prompts (text) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | text solutions/answers (rollouts) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper targets hard reasoning problems with explicit math problem examples and mentions coding as a covered reasoning domain. Inputs are textual problem prompts (sometimes paired with partial solutions), and outputs are textual rollouts with final answers. Based on the stated prompt/response limits and discussion of internal representations, the tasks operate over 1D token sequences with capped lengths and inferred static attention and constructed state; these dynamics are not otherwise specified directly.

## Evidence
### Task: Mathematical problem solving (reasoning)
- "Let  $k \geq 2$  be an integer. Find the smallest integer  $n \geq k+1$" (Appendix G)
- "You are given a problem and a partial solution." (Section 4, POPE System Instruction)
- "we assume that the rollout  $\mathbf{y}$  represents the final answer in a \boxed{} block." (Section 2, Preliminaries and Notation)
- Inference: Treated input/output as token sequences with capped lengths based on "max prompt length       | 2048" and "max response length     | 16384" (Section E.2, Table 4). Inferred constructed state from "the internal representation induced by a partial sequence" and static attention from fixed prompt length (Section 5.1; Section E.2).

### Task: Coding problem solving (reasoning)
- "domains such as math and coding." (Introduction)
- "To construct the hard problem set, we select problems from [49], OmniMath (levels 5-8) [16], and AceReason [9]." (Experimental setup)
- "Acereason-nemotron: Advancing math and code reasoning through reinforcement learning." (References)
- "You are given a problem and a partial solution." (Section 4, POPE System Instruction)
- "we assume that the rollout  $\mathbf{y}$  represents the final answer in a \boxed{} block." (Section 2, Preliminaries and Notation)
- Inference: Treated input/output as token sequences with capped lengths based on "max prompt length       | 2048" and "max response length     | 16384" (Section E.2, Table 4). Inferred constructed state from "the internal representation induced by a partial sequence" and static attention from fixed prompt length (Section 5.1; Section E.2).
