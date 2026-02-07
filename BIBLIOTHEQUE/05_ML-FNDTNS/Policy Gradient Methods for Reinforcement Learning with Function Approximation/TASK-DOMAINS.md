# Policy Gradient Methods for Reinforcement Learning with Function Approximation (Not specified in the paper)
Source: Policy Gradient Methods for Reinforcement Learning with Function Approximation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (action selection policy) | states s_t | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | action selection probabilities | 1D (t) (inferred) | Open (inferred) |
| prediction (action-value / advantage estimation) | state-action pairs (s, a) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | scalar value estimates Q^pi(s,a) / f_w(s,a) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper addresses reinforcement learning in MDPs with a stochastic policy, covering a control task that maps time-indexed states to action probabilities. It also includes value/advantage estimation via a learned function approximator that maps state-action pairs to scalar value estimates. The control interaction is inferred to be temporal and open-ended, while the value estimation is inferred to be a fixed-size, point-like prediction; attention and state dynamics are not explicitly discussed and are inferred from the described mappings.

## Evidence
### Task: control (action selection policy)
- "The state, action, and reward at each time  $t \in \{0, 1, 2, ...\}$  are denoted  $s_t$ ,  $a_t$ , and  $r_t$  respectively." (Section 1 Policy Gradient Theorem)
- "For example, the policy might be represented by a neural network whose input is a representation of the state, whose output is action selection probabilities" (Introduction, before Section 1)
- "$\pi(s, a, \theta) = Pr\left\{a_t = a \middle| s_t = s, \theta\right\}$" (Section 1 Policy Gradient Theorem)
- Inference: Labeled 1D (t) and Open because states/actions are indexed by time $t \in \{0, 1, 2, ...\}$ in an ongoing interaction; labeled Static/Direct because the policy is described as mapping a state representation directly to action selection probabilities with no runtime selection or memory described.

### Task: prediction (action-value / advantage estimation)
- "In the average reward formulation, the value of a state-action pair given a policy is defined as" (Section 1 Policy Gradient Theorem)
- "Let  $f_w: \mathcal{S} \times \mathcal{A} \to \Re$  be our approximation to  $Q^{\pi}$" (Section 2 Policy Gradient with Approximation)
- Inference: Labeled 0D and Fixed because $f_w$ maps a single state-action pair to a scalar value; labeled Static/Direct because no dynamic attention or constructed memory is described for the value estimator.
