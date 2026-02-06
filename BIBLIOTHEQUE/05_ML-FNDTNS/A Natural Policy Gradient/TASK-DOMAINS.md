# A Natural Policy Gradient (Not specified in the paper.)
Source: A Natural Policy Gradient.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (linear quadratic regulator) | state x(t) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | control signal u | 1D (t) (inferred) | Open (inferred) |
| control (2-state MDP) | state s (2-state MDP) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | actions (self- and cross-transition actions) | 1D (t) (inferred) | Open (inferred) |
| game playing / control (Tetris) | Tetris game state features (column heights, holes) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | actions a | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper frames tasks as policy optimization/control in MDPs and evaluates the method on simple MDPs (including an LQG regulator and a 2-state MDP) and the game of Tetris. The inputs are states (or state-derived features) and outputs are actions/control signals, with sequential decision-making implied by time-indexed state/action notation and infinite-horizon reward definitions. Dimensions and dynamics are therefore treated as 1D (t) with open horizons, while attention and state are inferred as static/direct because the policy is defined as a direct mapping from state to action.

## Evidence
### Task: control (linear quadratic regulator)
- "a simple 1-dimensional linear quadratic regulator with dynamics  $x(t+1) = .7x(t) + u(t) + \epsilon(t)$ ." (Section 5 Experiments)
- "The goal is to apply a control signal u to keep the system at x=0" (Section 5 Experiments)
- "stochastic policy  $\pi(a; s)$ , which is the probability of taking action a in state s" (Section 2 A Natural Gradient)
- " $s_t$  and  $a_t$  are the state and action at time t." (Section 2 A Natural Gradient)
- "\sum_{t=0}^{\infty} R(s_t,a_t)" (Section 2 A Natural Gradient)
- Inference: In/Out Dimension set to 1D (t) and Dynamics set to Open because states/actions are time-indexed (x(t+1), s_t, a_t) and rewards are defined over an infinite horizon; Attention/State set to Static/Direct because the policy is a direct state-to-action mapping with no attention or memory described.

### Task: control (2-state MDP)
- "a simple 2-state MDP (Figure 1B), which has self- and cross-transition actions" (Section 5 Experiments)
- "stochastic policy  $\pi(a; s)$ , which is the probability of taking action a in state s" (Section 2 A Natural Gradient)
- " $s_t$  and  $a_t$  are the state and action at time t." (Section 2 A Natural Gradient)
- "\sum_{t=0}^{\infty} R(s_t,a_t)" (Section 2 A Natural Gradient)
- Inference: In/Out Dimension set to 1D (t) and Dynamics set to Open because the MDP is defined over time-indexed states/actions with infinite-horizon reward; Attention/State set to Static/Direct because the policy is defined as a direct mapping from state to action with no attention or memory described.

### Task: game playing / control (Tetris)
- "The game of Tetris provides a challenging high dimensional problem." (Section 5 Experiments)
- "the heights of each column, the differences in height between adjacent columns, the maximum height, and the number of 'holes'." (Section 5 Experiments)
- "stochastic policy  $\pi(a; s)$ , which is the probability of taking action a in state s" (Section 2 A Natural Gradient)
- " $s_t$  and  $a_t$  are the state and action at time t." (Section 2 A Natural Gradient)
- "\sum_{t=0}^{\infty} R(s_t,a_t)" (Section 2 A Natural Gradient)
- Inference: In/Out Dimension set to 1D (t) and Dynamics set to Open because Tetris is treated as an MDP with time-indexed states/actions and infinite-horizon reward definition; Attention/State set to Static/Direct because the policy is defined as a direct mapping from state to action with no attention or memory described.
