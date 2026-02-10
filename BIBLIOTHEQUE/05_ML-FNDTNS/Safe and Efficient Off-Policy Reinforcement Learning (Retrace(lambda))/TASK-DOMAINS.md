# Safe and efficient off-policy reinforcement learning (Not specified in the paper)
Source: Safe and Efficient Off-Policy Reinforcement Learning (Retrace(lambda)).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Policy evaluation | State-action-reward trajectories from behaviour policy μ | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Action-value function Q^π over state-action pairs | 0D (inferred) | Fixed (inferred) |
| Control | Sample trajectories with behaviour policies and current Q-functions | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Optimal action-value function Q* and increasingly greedy target policies | 0D (inferred) | Fixed (inferred) |

## Summary
The paper explicitly covers two off-policy reinforcement learning tasks: policy evaluation and control. In both, the algorithm learns from temporally ordered trajectories, so the input is treated as 1D (t), and interaction is treated as open-ended because trajectories are represented as continuing sequences. The method maintains and updates Q-functions and policy sequences, supporting Constructed state and Dynamic attention as inferred from policy-driven action selection. Outputs are action-value/policy objects over a predefined state-action space, mapped here to 0D and Fixed dynamics by inference.

## Evidence
### Task: Policy evaluation
- "In the *policy evaluation* setting, we are given a fixed policy  $\pi$  whose value  $Q^{\pi}$  we wish to estimate from sample trajectories drawn from a behaviour policy  $\mu$ ." (Section 2 Off-Policy Algorithms)
- "We will consider trajectories of the form: $$x_0 = x, a_0 = a, r_0, x_1, a_1, r_1, x_2, a_2, r_2, \dots$$" (Section 1 Notation)
- Inference: `In Dimension = 1D (t)` and `In Dynamics = Open` are inferred from trajectory-form sequential input and unbounded continuation notation ("...," and sums over t) (Section 1 Notation; Section 2 Eq. (3)). `Attention Dynamic = Dynamic` is inferred because actions are selected by policies at runtime ("$a_t \sim \mu(\cdot|x_t)$"). `State Dynamic = Constructed` is inferred because the method iteratively constructs Q estimates (e.g., operator updates to Q and fixed point $Q^{\pi}$). `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from "A Q-function Q maps each state-action pair (x,a) to a value in  $\mathbb{R}$" over a defined state/action space (Section 1 Notation).

### Task: Control
- "In the *control* setting, we consider a sequence of policies that depend on our own sequence of Q-functions (such as  $\varepsilon$ -greedy policies), and seek to approximate  $Q^*$ ." (Section 2 Off-Policy Algorithms)
- "Then  $Q_k \to Q^*$  a.s." (Section 3.3 Online algorithms, Theorem 3)
- Inference: `In Dimension = 1D (t)` and `In Dynamics = Open` are inferred from sampled trajectories evolving over time (Section 3.3, "$x_0, a_0, r_0, x_1, a_1, r_1, \ldots$"). `Attention Dynamic = Dynamic` is inferred from runtime policy-dependent action generation and increasingly greedy policy updates (Sections 2 and 3.2). `State Dynamic = Constructed` is inferred because control uses and updates a constructed sequence of Q-functions and policies (Sections 2 and 3.2). `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from optimization over defined state-action spaces toward $Q^*$ (Section 1 Notation; Section 3.2).
