# Maximum a Posteriori Policy Optimisation (Not specified in the paper)
Source: Maximum a Posteriori Policy Optimisation (MPO).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (continuous action) | states s (continuous) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | actions a (Gaussian policy) | 1D (t) (inferred) | Open (inferred) |
| control (discrete action) | states s | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | actions a (categorical policy) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper presents an off-policy reinforcement learning control method evaluated on continuous control domains (DeepMind Control Suite and parkour) and on discrete control Atari/ALE games. It formulates problems as MDP trajectories of states and actions over time, with returns defined over potentially unbounded horizons. Policies map states to action distributions (Gaussian for continuous actions and categorical for discrete actions), and the method learns a Q-function, indicating constructed internal state while observation selection remains fixed by the given state.

## Evidence
### Task: control (continuous action)
- "we start by looking at the continuous control tasks of the DeepMind Control Suite" (Section 5 Experiments)
- "In both cases we use a Gaussian distribution for the policy" (Section 5 Experiments)
- "The MDP consists of: continuous states s, actions a" (Section 2.2 Markov Decision Processes)
- "trajectory  $\tau_{\pi} = \{(s_0, a_0) \dots (s_T, a_T)\}$" (Section 2.2 Markov Decision Processes)
- "sum_{t=0}^{\infty} \gamma^t r(s_t, s_t)" (Section 2.2 Markov Decision Processes)
- "specify a probability distribution over action choices given any state" (Section 2.2 Markov Decision Processes)
- "obtain a parametric representation of the Q-function" (Section 4 Policy Evaluation)
- Inference: Marked In/Out Dimension as 1D (t) and Dynamics as Open because trajectories are time-indexed and returns sum to infinity; marked Attention as Static because the policy conditions on a given state; marked State as Constructed because a Q-function is learned with a neural network.

### Task: control (discrete action)
- "initial experiments for discrete control using ATARI environments using a categorical policy distribution" (Section 5 Experiments)
- "subset of the games contained contained in the "Arcade Learning Environment" (ALE)" (Appendix B Additional Experiment: Discrete Control)
- "trajectory  $\tau_{\pi} = \{(s_0, a_0) \dots (s_T, a_T)\}$" (Section 2.2 Markov Decision Processes)
- "sum_{t=0}^{\infty} \gamma^t r(s_t, s_t)" (Section 2.2 Markov Decision Processes)
- "specify a probability distribution over action choices given any state" (Section 2.2 Markov Decision Processes)
- "obtain a parametric representation of the Q-function" (Section 4 Policy Evaluation)
- Inference: Marked In/Out Dimension as 1D (t) and Dynamics as Open because trajectories are time-indexed and returns sum to infinity; marked Attention as Static because the policy conditions on a given state; marked State as Constructed because a Q-function is learned with a neural network.