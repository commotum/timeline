# Proximal Policy Optimization Algorithms (Not specified in the paper.)
Source: Proximal Policy Optimization Algorithms.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (simulated robotic locomotion / humanoid control) | Environment states/observations s_t over timesteps | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Actions a_t over timesteps | 1D (t) (inferred) | Capped (inferred) |
| Control (Atari game playing) | Environment states/observations s_t over timesteps | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Actions a_t over timesteps | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates PPO on control tasks in simulated robotics (MuJoCo benchmarks and Roboschool humanoid problems) and on Atari game playing. The interaction is described via state-action sequences over timesteps, which supports temporal 1D inputs/outputs and capped horizon segments per update (inferred from fixed T timesteps). The paper does not explicitly specify attention dynamics or constructed state beyond the raw observations.

## Evidence
### Task: Control (simulated robotic locomotion / humanoid control)
- "including simulated robotic locomotion" (Abstract)
- "we used 7 simulated robotics tasks implemented in OpenAI Gym [Bro+16], which use the MuJoCo [TET12] physics engine." (Section 6.1)
- "we train on a set of problems involving a 3D humanoid, where the robot must run, steer, and get up off the ground" (Section 6.3)
- "log \pi_\theta(a_t \mid s_t)" (Section 2.1, Eq. 2)
- "Run policy \pi_{\theta_{\text{old}}} in environment for T timesteps" (Algorithm 1, Section 5)
- Inference: The tasks are framed as temporal state-action sequences with fixed-length segments of T timesteps, so In/Out Dimension = 1D (t) and In/Out Dynamics = Capped (supported by the state-action form and the fixed T timesteps quote above).

### Task: Control (Atari game playing)
- "including simulated robotic locomotion and Atari game playing" (Abstract)
- "We also ran PPO on the Arcade Learning Environment [Bel+15] benchmark" (Section 6.4)
- "log \pi_\theta(a_t \mid s_t)" (Section 2.1, Eq. 2)
- "Run policy \pi_{\theta_{\text{old}}} in environment for T timesteps" (Algorithm 1, Section 5)
- Inference: The tasks are framed as temporal state-action sequences with fixed-length segments of T timesteps, so In/Out Dimension = 1D (t) and In/Out Dynamics = Capped (supported by the state-action form and the fixed T timesteps quote above).
