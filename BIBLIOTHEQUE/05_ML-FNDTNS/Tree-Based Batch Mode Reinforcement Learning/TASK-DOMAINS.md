# Tree-Based Batch Mode Reinforcement Learning (Not specified in the paper)
Source: Tree-Based Batch Mode Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (batch-mode reinforcement learning for optimal policy learning) | State-action-reward-next-state four-tuples (transitions) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Stationary control policy and per-state action selection | 0D (inferred) | Open (inferred) |

## Summary
The paper covers reinforcement learning as a control task, where policies are learned from batches of transition tuples. The input is explicitly temporal transition data indexed by time steps, and the output is a control policy that emits actions for states. The paper supports open-ended interaction over an infinite horizon and arbitrary-size transition sets, while the Attention and State labels are inferred from the described fitted Q-iteration procedure.

## Evidence
### Task: Control (batch-mode reinforcement learning for optimal policy learning)
- "Reinforcement learning aims to determine an optimal control policy from interaction with a system or from observations gathered from a system." (Abstract)
- "In this paper we consider batch mode learning, where the learning agent is in principle not directly interacting with the system but receives only a set of four-tuples and is asked to determine from this set a control policy which is as close as possible to an optimal policy." (Section 1)
- "When the stopping conditions - whatever they are - are reached, the final control policy, seen as an approximation of the optimal stationary closed loop control policy is derived by
$$\hat{\mu}_N^*(x) = \underset{u \in U}{\arg\max} \hat{Q}_N(x, u).$$" (Section 3.4)
- Inference: In Dimension is marked as 1D (t) because the core data object is the transition tuple $(x_t, u_t, r_t, x_{t+1})$ indexed by time step $t$ and gathered from episodes (Sections 1 and 2.1). In/Out Dynamics are marked Open because the method targets "an infinite horizon" control objective and allows "a set of transitions of arbitrary size" (Sections 2 and 2.1). Attention Dynamic is marked Static because the policy/regression interface consumes predefined state-action variables rather than runtime retrieval over external context (Section 3.1). State Dynamic is marked Constructed because the algorithm iteratively builds internal $\hat{Q}_N$ approximations and derives policy from them (Section 3.1 and Section 3.4). Out Dimension is marked 0D because action selection is produced per queried state via $\arg\max_u \hat{Q}_N(x,u)$ (Section 3.4).
