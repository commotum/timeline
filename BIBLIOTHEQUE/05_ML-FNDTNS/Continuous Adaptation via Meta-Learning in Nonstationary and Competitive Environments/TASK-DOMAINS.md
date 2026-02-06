# Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments (Not specified in the paper.)
Source: Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Locomotion control (nonstationary environment) | Body position/velocity and leg angles/velocities observations | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Joint torques (actions) | 1D (t) (inferred) | Capped (inferred) |
| Competitive control (RoboSumo iterated adaptation games) | Self and opponent positions, joint angles/velocities, and forces observations | 1D (t) (inferred) | Fixed | Static (inferred) | Constructed (inferred) | Continuous actions | 1D (t) (inferred) | Fixed |

## Summary
The paper covers continuous-adaptation reinforcement-learning control in two domains: nonstationary single-agent locomotion and competitive RoboSumo iterated adaptation games. Both domains use continuous proprioceptive/force observations and continuous action outputs, which implies temporal (1D (t)) sequences. RoboSumo episodes are explicitly fixed-length (500 time steps), while locomotion is described episodically without a stated step count; attention and state dynamics are inferred as static observation interfaces with constructed internal state via recurrent policies.

## Evidence
### Task: Locomotion control (nonstationary environment)
- "First, we consider the problem of robotic locomotion in a changing environment." (Section 4.1 Dynamic)
- "observes the absolute position and velocity of its body, the angles and velocities of its legs, and it acts by applying torques to its joints." (Section 4.1 Dynamic)
- "linearly changes from 1 to 0 over the course of 7 episodes." (Section 4.1 Dynamic)
- "trajectory,  $\boldsymbol{\tau} := (\mathbf{x}_0, \mathbf{a}_1, \mathbf{x}_1, R_1, \dots, \mathbf{a}_H, \mathbf{x}_H, R_H)" (Section 3.1)
- "The state in LSTM-based architectures was kept throughout each episode and reset to zeros at the beginning of each new episode." (Appendix B)
- Inference: The input/output are treated as time-indexed sequences (1D (t)) based on the trajectory definition and episodic interaction; the interaction length is bounded but not fixed in steps for locomotion, so dynamics are marked Capped (inferred). Attention is Static (inferred) because the observation fields are fixed, and State is Constructed (inferred) due to recurrent state being kept across episodes.

### Task: Competitive control (RoboSumo iterated adaptation games)
- "Our multi-agent environment, RoboSumo, allows agents to compete in the 1-vs-1 regime" (Section 4.2 Competitive)
- "each agent observes positions of itself and the opponent, its own joint angles, the corresponding velocities, and the forces" (Section 4.2 Competitive)
- "The action spaces are continuous." (Section 4.2 Competitive)
- "fixed length episodes (500 time steps each)" (Section 4.2 Competitive)
- "The state in LSTM-based architectures was kept throughout each episode and reset to zeros at the beginning of each new episode." (Appendix B)
- Inference: The observation/action streams are time-indexed (1D (t)) based on fixed-length episodes; Attention is Static (inferred) because the observation fields are predefined, and State is Constructed (inferred) due to recurrent state maintained across interaction.
