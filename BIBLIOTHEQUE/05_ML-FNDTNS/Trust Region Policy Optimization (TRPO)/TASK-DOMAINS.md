# Trust Region Policy Optimization (2015)
Source: Trust Region Policy Optimization (TRPO).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (simulated robotic locomotion) | Robot state vectors (generalized positions and velocities) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Joint-torque controls/actions | 0D (inferred) | Fixed (inferred) |
| Control (Atari game playing from images) | Raw game screen images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Discrete game actions | 0D (inferred) | Fixed (inferred) |

## Summary
The paper applies TRPO to control tasks in two input modalities: low-dimensional robot state vectors and raw image observations from Atari games. Across both domains, the model is described as a policy mapping observations to actions with fixed per-step interfaces (state vectors or preprocessed images to action distributions), rather than variable-length inputs. The OCR supports static runtime observation handling and direct state-to-action mapping, with no explicit retrieval mechanism or persistent internal memory process described.

## Evidence
### Task: Control (simulated robotic locomotion)
- "Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits" (Abstract)
- "The states of the robots are their generalized positions and velocities, and the controls are joint torques." (Section 8.1 Simulated Robotic Locomotion)
- "Swimmer. 10-dimensional state space" (Section 8.1 Simulated Robotic Locomotion)
- "The policy, which is a conditional probability distribution  $\pi_{\theta}(a|s)$ , can be parameterized with a neural network. This neural network maps (deterministically) from the state vector s to a vector  $\mu$" (Appendix D)
- Inference: `0D`/`Fixed` labels are inferred from fixed-dimensional state and control vectors (e.g., "10-dimensional state space" and explicit control variables). `Static` attention and `Direct` state are inferred because the described policy maps current state vectors directly to action distributions without an explicit runtime selection/retrieval mechanism or persistent constructed memory.

### Task: Control (Atari game playing from images)
- "we trained policies for playing Atari games, using raw images as input." (Section 8.2 Playing Games from Images)
- "the policy was represented by the convolutional neural network shown in Figure 3" (Section 8.2 Playing Games from Images)
- "Our algorithms (bottom rows) were run once on each task, with the same architecture and parameters." (Table 1 caption, Section 8.2)
- "For the experiments with discrete actions (Atari), we use a factored discrete action space" (Appendix D)
- Inference: `2D (x, y)` is inferred from "raw images" as spatial grids; `Fixed` input/output dynamics are inferred from fixed architecture/parameterization and discrete action-space specification. `Static` attention and `Direct` state are inferred because the policy is described as a direct neural mapping from observation input to action distribution, with no explicit dynamic retrieval or maintained internal memory state.
