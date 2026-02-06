# Deep Reinforcement Learning from Human Preferences (Not specified in the paper)
Source: Deep reinforcement learning from human preferences.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (Atari game playing) | Atari observations (stacked frames) | 3D (x, y, t) | Fixed | Not specified in the paper. | Not specified in the paper. | Actions (game controls) | 0D (inferred) | Fixed (inferred) |
| Control (MuJoCo robotics) | Observations (o_t) from the environment | 0D (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Actions (robot control) | 0D (inferred) | Fixed (inferred) |
| Reward prediction (Atari) | Observations and actions; 84x84 images with 4-frame stacks | 3D (x, y, t) | Fixed | Not specified in the paper. | Not specified in the paper. | Scalar reward estimate \hat{r}(o_t, a_t) | 0D | Fixed |
| Reward prediction (MuJoCo) | Observations and actions (o_t, a_t) | 0D (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Scalar reward estimate \hat{r}(o_t, a_t) | 0D | Fixed |

## Summary
The paper covers sequential control in Atari games and simulated MuJoCo robotics, and it also trains reward predictors from preference data to supply scalar rewards. Atari inputs are explicitly 84x84 images stacked across 4 frames, supporting a 3D (x, y, t) fixed input for both policy and reward prediction, while reward outputs are scalar. For MuJoCo, the text describes feature-based rewards over positions/velocities and an MLP reward predictor, which implies fixed-size vector inputs (0D inferred). Attention and state dynamics are not specified.

## Evidence
### Task: Control (Atari game playing)
- "Atari games in the Arcade Learning Environment (Bellemare et al., 2013), and robotics tasks in the physics simulator MuJoCo (Todorov et al., 2012)." (Section 1 Introduction)
- "at each time t the agent receives an observation  $o_t \in \mathcal{O}$  from the environment and then sends an action  $a_t \in \mathcal{A}$" (Section 2.1)
- "stacking of 4 frames" (Section A.2)
- Inference: Out Dimension and Out Dynamics are treated as a single fixed action per step based on the definition of action  $a_t \in \mathcal{A}$. (Section 2.1)

### Task: Control (MuJoCo robotics)
- "Atari games in the Arcade Learning Environment (Bellemare et al., 2013), and robotics tasks in the physics simulator MuJoCo (Todorov et al., 2012)." (Section 1 Introduction)
- "at each time t the agent receives an observation  $o_t \in \mathcal{O}$  from the environment and then sends an action  $a_t \in \mathcal{A}$" (Section 2.1)
- "The reward functions in these tasks are linear functions of distances, positions and velocities" (Section 3.1.1)
- Inference: In Dimension and In Dynamics are inferred as fixed-size vector observations (0D) from the feature-based description (distances/positions/velocities). (Section 3.1.1)
- Inference: Out Dimension and Out Dynamics are treated as a single fixed action per step based on the definition of action  $a_t \in \mathcal{A}$. (Section 2.1)

### Task: Reward prediction (Atari)
- "a reward function estimate  $\hat{r}: \mathcal{O} \times \mathcal{A} \to \mathbb{R}$" (Section 2.2)
- "we use 84x84 images as inputs (the same as the inputs to the policy)" (Section A.2)
- "stack 4 frames for a total 84x84x4 input tensor." (Section A.2)

### Task: Reward prediction (MuJoCo)
- "a reward function estimate  $\hat{r}: \mathcal{O} \times \mathcal{A} \to \mathbb{R}$" (Section 2.2)
- "The reward predictor is a two-layer neural network with 64 hidden units each" (Section A.1)
- "The reward functions in these tasks are linear functions of distances, positions and velocities" (Section 3.1.1)
- Inference: In Dimension and In Dynamics are inferred as fixed-size vector inputs (0D) from the MLP reward predictor and feature-based reward description. (Sections A.1 and 3.1.1)
