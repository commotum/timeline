# Policy Distillation (Not specified in the paper)
Source: Policy Distillation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (Atari gameplay) | observation sequence of images (pixel frames) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions (discrete) | 0D | Fixed (inferred) |

## Summary
The paper centers on control policies for Atari games that map pixel observations to discrete actions, with DQN and distilled students trained on image-based gameplay data. Inputs are image frames taken over short consecutive windows, implying spatiotemporal visual inputs, while outputs are discrete action choices drawn from a fixed action set. The described systems operate with fixed-size observation windows, static attention to that window, and direct (reactive) state derived from the observation sequence rather than an explicit external memory.

## Evidence
### Task: control (Atari gameplay)
- "The deep Q-network (DQN) algorithm interacts with an environment, receiving pixel observations and rewards." (Section 1 Introduction)
- "At each step, an agent chooses the action that maximizes its predicted cumulative reward," (Section 1 Introduction)
- "a neural network is optimized to predict the average discounted future return of each possible action given a small number of consecutive observations." (Section 3.1 Deep Q-learning)
- "**actions**  $a_i \in \mathcal A = \{1,...,K\}$" (Section 3.1 Deep Q-learning)
- "The DQN teacher's outputs (Q-values for all actions) alongside the inputs (images) were held in a buffer." (Section 4.1 Training and Evaluation)
- Inference: Marked In Dimension as 3D (x, y, t) and In Dynamics/Attention as Fixed/Static because the model consumes a small number of consecutive image observations; marked Out Dynamics as Fixed because actions are from a finite action set (see action-set quote above). Marked State Dynamic as Direct because the policy is described as choosing actions from observations without an explicit constructed memory beyond the observation sequence (supported by the quotes above).
