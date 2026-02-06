# A Distributional Perspective on Reinforcement Learning (2017)
Source: A Distributional Perspective on Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (value distribution estimation) | state-action pairs (x,a) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | distribution over returns (value distribution); atom probabilities p_i(x,a) | Not specified in the paper. | Fixed (inferred) |
| control (action selection) | current state x | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | action / policy over actions | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper frames reinforcement learning around predicting value distributions for state-action pairs and using those estimates to select actions (control). Task descriptions stay at the level of states, actions, rewards, and return distributions rather than specifying concrete sensory modalities, so most dimension/dynamics/attention/state attributes are not explicitly defined. The categorical algorithm’s output is a fixed-size discrete distribution over N atoms, giving a fixed output interface for value-distribution prediction.

## Evidence
### Task: prediction (value distribution estimation)
- "we will view  $Z^{\pi}$  as a mapping from state-action pairs to distributions over returns, and call it the *value distribution*." (Section 3. The Distributional Bellman Operators)
- "The atom probabilities are given by a parametric model  $\theta: \mathcal{X} \times \mathcal{A} \to \mathbb{R}^N$" (Section 4.1. Parametric Distribution)
- "output the atom probabilities  $p_i(x,a)$  instead of action-values" (Section 5. Evaluation on Atari 2600 Games)
- Inference: Out Dynamics is Fixed because the value distribution is parameterized by N atoms (fixed-size probability vector). (Section 4.1. Parametric Distribution)

### Task: control (action selection)
- "at each step, the agent selects an action based on its current state" (Section 2. Setting)
- "A stationary policy  $\pi$  maps each state  $x \in \mathcal{X}$  to a probability distribution over the action space  $\mathcal{A}$ ." (Section 2. Setting)
