# HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION (Not specified in the paper)
Source: High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cart-pole balancing (control) | state (s_t) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | actions (a_t) | 1D (t) (inferred) | Capped (inferred) |
| 3D bipedal locomotion (control) | raw kinematics | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | joint torques | 1D (t) (inferred) | Capped (inferred) |
| 3D quadrupedal locomotion (control) | raw kinematics | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | joint torques | 1D (t) (inferred) | Capped (inferred) |
| 3D biped standing up / getting up (control) | raw kinematics | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | joint torques | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates reinforcement-learning control on cart-pole balancing and three 3D robot behaviors: bipedal locomotion, quadrupedal locomotion, and biped standing up/getting up. Inputs/outputs are described as state or raw kinematics mapped to actions or joint torques, with explicit episode limits of 1000 or 2000 timesteps. Based on the trajectory formulation and episode caps, the tasks are characterized as 1D (t) sequences with capped dynamics; attention and state dynamics are not explicitly specified and are marked as inferred Static/Direct.

## Evidence
### Task: Cart-pole balancing (control)
- "We evaluated our approach on the classic cart-pole balancing problem" (Section 6.2 Experimental Setup)
- "A trajectory  $(s_0, a_0, s_1, a_1, \dots)$" (Preliminaries)
- "sampling actions according to the policy  $a_t \sim \pi(a_t \mid s_t)$" (Preliminaries)
- "with a maximum length of 1000 timesteps" (Section 6.2.2 Task Details)
- Inference: 1D (t) and Capped are inferred from "A trajectory  $(s_0, a_0, s_1, a_1, \dots)$" and "maximum length of 1000 timesteps"; Static/Direct from "policy  $a_t \sim \pi(a_t \mid s_t)$".

### Task: 3D bipedal locomotion (control)
- "3D biped locomotion" (Section 6.2.2 Task Details)
- "our neural network policies map directly from raw kinematics to joint torques." (Abstract)
- "Each episode was terminated after 2000 timesteps" (Section 6.2.2 Task Details)
- Inference: 1D (t) and Capped are inferred from "A trajectory  $(s_0, a_0, s_1, a_1, \dots)$" and "Each episode was terminated after 2000 timesteps"; Static/Direct from "map directly from raw kinematics to joint torques."

### Task: 3D quadrupedal locomotion (control)
- "Quadruped locomotion" (Section 6.2.2 Task Details)
- "our neural network policies map directly from raw kinematics to joint torques." (Abstract)
- "Each episode was terminated after 2000 timesteps" (Section 6.2.2 Task Details)
- Inference: 1D (t) and Capped are inferred from "A trajectory  $(s_0, a_0, s_1, a_1, \dots)$" and "Each episode was terminated after 2000 timesteps"; Static/Direct from "map directly from raw kinematics to joint torques."

### Task: 3D biped standing up / getting up (control)
- "dynamically standing up, for the biped, which starts off laying on its back" (Section 6.2 Experimental Setup)
- "our neural network policies map directly from raw kinematics to joint torques." (Abstract)
- "Each episode was terminated after 2000 timesteps" (Section 6.2.2 Task Details)
- Inference: 1D (t) and Capped are inferred from "A trajectory  $(s_0, a_0, s_1, a_1, \dots)$" and "Each episode was terminated after 2000 timesteps"; Static/Direct from "map directly from raw kinematics to joint torques."
