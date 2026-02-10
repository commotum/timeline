# Technical Note Q-Learning (Not specified in the paper.)
Source: Technical Note- Q-learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (optimal policy learning) | Episode stream of state-action-next-state-reward tuples `(x_n, a_n, y_n, r_n)` | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Optimal action-values `Q*(x, a)` | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper covers a single reinforcement-learning control task: learning to act optimally in a controlled Markov process. The modeled interaction is a temporal episode stream of state, action, next-state, and reward observations, while the learned output is a discrete action-value table over state-action pairs. The OCR text supports an open input dynamic (infinite episode sampling) and a fixed output dynamic (finite, discrete look-up-table representation). Attention is classified as static and state as constructed because the update rule uses a fixed local transition tuple and maintains persistent learned `Q` values.

## Evidence
### Task: Control (optimal policy learning)
- "The task facing the agent is that of determining an optimal policy, one that maximizes total discounted expected reward." (Section 2. The task for Q-learning)
- "The object in Q-learning is to estimate the Q values for an optimal policy." (Section 2. The task for Q-learning)
- "In the nth episode, the agent: observes its current state  $x_n$ , selects and performs an action  $a_n$ , observes the subsequent state  $y_n$ , receives an immediate payoff  $r_n$" (Section 2. The task for Q-learning)
- "Note that this description assumes a look-up table representation for the  $Q_n(x, a)$ ." (Section 2. The task for Q-learning)
- "The most important condition implicit in the convergence theorem given below is that the sequence of episodes that forms the basis of learning must include an infinite number of episodes for each starting state and action." (Section 2. The task for Q-learning)
- Inference: `In Dimension = 1D (t)` is inferred from repeated "time step" and episode-sequence framing; `In Dynamics = Open` is inferred from the explicit requirement of an "infinite number of episodes"; `Out Dimension = 2D (x, y)` and `Out Dynamics = Fixed` are inferred from `Q_n(x, a)` with discrete finite states/actions and the stated look-up-table representation; `Attention Dynamic = Static` is inferred because each update uses a fixed local tuple `(x_n, a_n, y_n, r_n)` rather than runtime selection over variable context; `State Dynamic = Constructed` is inferred because learned `Q` values are explicitly updated and carried across episodes as reusable decision state.
