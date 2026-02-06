# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING (Not specified in the paper)
Source: Continuous Control with Deep Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous control (low-dimensional state observations) | low-dimensional state descriptions (e.g., joint angles, positions) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct | continuous actions (e.g., joint torques; acceleration/braking/steering) | 0D (inferred) | Fixed (inferred) |
| continuous control (pixel observations) | pixel renderings / stacked frames (RGB feature maps) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct | continuous actions (e.g., joint torques; acceleration/braking/steering) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper addresses continuous control policy learning across simulated physics tasks and car driving, using both low-dimensional state vectors and raw pixel observations. Inputs are either fixed-size state descriptions or stacked image renderings (64x64 with multiple frames), while outputs are continuous action vectors (e.g., joint torques or driving controls). This yields 0D and 3D (x, y, t) input dimensions with fixed-size dynamics, static attention, and direct state mapping (s_t = x_t).

## Evidence
### Task: continuous control (low-dimensional state observations)
- "robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving." (Abstract)
- "low-dimensional state description (such as joint angles and positions)" (Section 4 Results)
- "actions are real-valued  $a_t \in \mathbb{R}^N$ ." (Section 2 Background)
- "we assumed the environment is fully-observed so  $s_t = x_t$ ." (Section 2 Background)
- "maps states to a probability distribution over the actions" (Section 2 Background)
- Inference: In Dimension = 0D (inferred) and In Dynamics = Fixed (inferred) because the input is a low-dimensional state description (joint angles/positions), implying a fixed-size state vector; Out Dimension = 0D (inferred) and Out Dynamics = Fixed (inferred) because actions are real-valued vectors; Attention Dynamic = Static (inferred) because the policy maps states to actions without any described runtime selection. (supporting quotes above)

### Task: continuous control (pixel observations)
- "directly from raw pixel inputs." (Abstract)
- "the observation reported to the agent contains 9 feature maps (the RGB of each of the 3 renderings)" (Section 4 Results)
- "The frames were downsampled to 64x64 pixels" (Section 4 Results)
- "actions are real-valued  $a_t \in \mathbb{R}^N$ ." (Section 2 Background)
- "we assumed the environment is fully-observed so  $s_t = x_t$ ." (Section 2 Background)
- "maps states to a probability distribution over the actions" (Section 2 Background)
- Inference: In Dimension = 3D (x, y, t) (inferred) and In Dynamics = Fixed (inferred) because each observation is a fixed-size stack of 3 renderings (9 RGB feature maps) at 64x64 resolution; Out Dimension = 0D (inferred) and Out Dynamics = Fixed (inferred) because actions are real-valued vectors; Attention Dynamic = Static (inferred) because the policy maps states to actions without any described runtime selection. (supporting quotes above)
