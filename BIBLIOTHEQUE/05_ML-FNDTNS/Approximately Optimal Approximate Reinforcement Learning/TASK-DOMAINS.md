# Approximately Optimal Approximate Reinforcement Learning (Not specified in the paper.)
Source: Approximately Optimal Approximate Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| policy optimization (approximate optimal policy) | state trajectories from an MDP (s_t; s) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | actions / policy π(a;s) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper addresses reinforcement learning via conservative policy iteration to obtain an approximately optimal policy in finite MDPs. It characterizes time-indexed state and action sequences and uses infinite-horizon discounted returns, which supports 1D temporal inputs/outputs with open-ended dynamics. The policy is defined over provided states, while value/advantage functions and their approximation are used to improve the policy, indicating static attention and constructed state (both inferred from the text).

## Evidence
### Task: policy optimization (approximate optimal policy)
- "we present the conservative policy iteration algorithm which finds an \"approximately\" optimal policy" (Abstract)
- "a stochastic policy π(a;s), which is the probability of taking action a in state s" (Section 2 Preliminaries)
- "$s_t$  and  $a_t$  are random variables for the state and action at time t" (Section 2 Preliminaries)
- "\sum_{t=0}^{\infty} \gamma^{t} \mathcal{R}(s_{t}, a_{t})" (Section 2 Preliminaries)
- "This greedy policy chooser can be implemented using standard value function approximation techniques." (Abstract)
- Inference: Mapped time-indexed sequences ($s_t$, $a_t$) and infinite-horizon return to 1D (t) and Open dynamics; treated the policy as Static attention because it conditions on provided states; marked State Dynamic as Constructed because the algorithm relies on value/advantage function approximation for policy improvement.
