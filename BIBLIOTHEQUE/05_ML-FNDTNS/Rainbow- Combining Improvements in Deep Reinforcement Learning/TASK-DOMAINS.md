# Rainbow: Combining Improvements in Deep Reinforcement Learning (2018)
Source: Rainbow- Combining Improvements in Deep Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (Atari game playing) | stack of raw pixel frames (observations/state) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions (discrete action selection) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers reinforcement-learning control for playing Atari 2600 games from raw pixel observations. Inputs are fixed-size stacks of image frames, and outputs are discrete actions selected at each time step. Based on the described interface, the task uses static attention over the provided observation and a direct (reactive) state, with fixed input and output dynamics.

## Evidence
### Task: control (Atari game playing)
- "learn, from raw pixels, how to play many Atari games at human-level performance." (Introduction)
- "the environment provides the agent with an observation  $S_t$ , the agent responds by selecting an action  $A_t$" (Background)
- "fed as input to the network in the form of a stack of raw pixel frames)." (Deep reinforcement learning and DQN)
- "Observations are grey-scaled and rescaled to  $84 \times 84$  pixels." (Appendix, Table 3)
- "4 consecutive frames are concatenated as each state's representation." (Appendix, Table 3)
- "$\mathcal{A}$  is a finite set of actions" (Background)
- "The output layer of the network has a number of units equal to the number of actions available in the game." (Appendix, Table 4)
- Inference: In Dimension = 3D (x, y, t) and In Dynamics = Fixed because the state is a fixed-size stack of consecutive frames at 84x84 resolution; Attention Dynamic = Static and State Dynamic = Direct because action selection uses the provided observation $S_t$ without any described adaptive input selection or explicit memory; Out Dimension = 0D and Out Dynamics = Fixed because the agent selects a single action from a finite action set. (Supported by the quotes above.)
