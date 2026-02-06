# Deep Reinforcement Learning with Double Q-learning (2016)
Source: Deep Reinforcement Learning with Double Q-learning (Double DQN).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (policy learning) | screen pixels (last four frames) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action values (Q-values) for each action | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper addresses value-based reinforcement learning for game-playing control in the Atari 2600 domain, learning policies from raw screen pixels. The described setup uses a fixed stack of four frames as input and produces a fixed-size vector of action values, implying a 3D (x, y, t) input and 1D output with fixed dynamics (inferred). Attention is static and the decision state is direct (inferred) because the model consumes a fixed observation window without any dynamic input selection or explicit memory described.

## Evidence
### Task: control (policy learning)
- "The goal of reinforcement learning (Sutton and Barto, 1998) is to learn good policies for sequential decision problems, by optimizing a cumulative future reward signal." (Section: Introduction, before Background)
- "The goal is for a single algorithm, with a fixed set of hyperparameters, to learn to play each of the games separately from interaction given only the screen pixels as input." (Section: Empirical results)
- "The network takes the last four frames as input and outputs the action value of each action." (Section: Empirical results)
- "A deep Q network (DQN) is a multi-layered neural network that for a given state s outputs a vector of action values Q(s,·;θ)" (Section: Deep Q Networks)
- Inference: Input dimension and dynamics are 3D (x, y, t) and Fixed because the input is explicitly "the last four frames"; output dimension and dynamics are 1D and Fixed because the model "outputs a vector of action values" over a fixed action set; attention is Static and state is Direct because the model processes a fixed observation stack with no dynamic selection or explicit memory described. (inferred from the quotes above)
