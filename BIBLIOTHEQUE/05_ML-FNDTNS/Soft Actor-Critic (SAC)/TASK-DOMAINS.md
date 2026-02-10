# Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (Not specified in the paper)
Source: Soft Actor-Critic (SAC).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous control (maximum-entropy reinforcement learning) | continuous environment state \(\mathbf{s}_t\) in MDP transitions | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | continuous action \(\mathbf{a}_t\) from stochastic policy \(\pi(\mathbf{a}_t|\mathbf{s}_t)\) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers one primary task domain: continuous-control reinforcement learning with a stochastic actor in continuous state and action spaces. The operational interface is per-step state-to-action control, which supports a 0D (point-like) interpretation for both input and output objects, with fixed-size vectors per environment (inferred from the task setup and reported action dimensions). The policy is described as conditioning on the provided current state, which supports Static attention and Direct state usage in this classification.

## Evidence
### Task: continuous control (maximum-entropy reinforcement learning)
- "our method achieves state-of-the-art performance on a range of continuous control benchmark tasks" (Abstract)
- "We address policy learning in continuous action spaces." (Section 3.1)
- "We consider an infinite-horizon Markov decision process (MDP), defined by the tuple  $(\mathcal{S},\mathcal{A},p,r)$ , where the state space  $\mathcal{S}$  and the action space  $\mathcal{A}$  are continuous" (Section 3.1)
- "\mathbf{a}_t \sim \pi_{\phi}(\mathbf{a}_t|\mathbf{s}_t)" (Algorithm 1, Section 4.2)
- "We compare our method to prior techniques on a range of challenging continuous control tasks from the OpenAI gym benchmark suite" (Section 5)
- Inference: In/Out Dimension are marked 0D (inferred) because the policy is applied at each step as state \(\mathbf{s}_t\) to action \(\mathbf{a}_t\), i.e., point-like control decisions per timestep (Sections 3.1, 4.2). In/Out Dynamics are marked Fixed (inferred) because each environment instance has fixed action dimensionality (Table 2: 3, 6, 8, 17, 21). Attention Dynamic is Static (inferred) because the runtime policy interface is explicitly \(\pi(\mathbf{a}_t|\mathbf{s}_t)\) with no described runtime retrieval/selection mechanism (Sections 3.1, 4.2). State Dynamic is Direct (inferred) because control is defined directly over current environment state/action tuples in the MDP formulation and Algorithm 1 (Sections 3.1, 4.2).
