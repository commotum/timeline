# Recurrent Experience Replay in Distributed Reinforcement Learning (Not specified in the paper.)
Source: Recurrent Experience Replay in Distributed Reinforcement Learning (R2D2).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control / action selection (inferred) | Observation sequences (4-frame stacks on Atari; single RGB frames on DMLab), plus previous action and reward; language input for DMLab language tasks (inferred) | 2D (x, y) (inferred); 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Discrete actions (action set) | 0D (inferred); 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper presents R2D2 as a value-based RL agent that interacts with environments by receiving observations and selecting discrete actions on Atari-57 and DMLab-30. Inputs are visual observation sequences (4-frame stacks or single RGB frames) with previous action and reward, and some DMLab levels include language inputs via a language LSTM. From these descriptions, the task involves temporal, image-based inputs with capped episode lengths, static attention, and constructed recurrent state from the LSTM, producing discrete action choices over time.

## Evidence
### Task: Control / action selection (inferred)
- "Within this framework, the agent receives an observation  $o \in \Omega$" (Section 2.1 Reinforcement Learning)
- "When the agent takes an action  $a \in \mathcal{A}$  the environment responds" (Section 2.1 Reinforcement Learning)
- "Like Ape-X, we use 4-frame-stacks and the full 18-action set when training on Atari." (Section 2.3 The Recurrent Replay Distributed DQN Agent)
- "On DMLab, we use single RGB frames as observations" (Section 2.3 The Recurrent Replay Distributed DQN Agent)
- "let  $o_t, \ldots, o_{t+m}$  denote the replay sequence of observations" (Section 3 Training Recurrent RL Agents with Experience Replay)
- "denote by  $h_{t+1} = h(o_t, h_t; \\theta)$  and  $q(h_t; \\theta)$  the recurrent state" (Section 3 Training Recurrent RL Agents with Experience Replay)
- "Additionally, the LSTM receives as input the reward and one-hot action vector from the previous time step." (Hyper-Parameters)
- "a^* = \arg\max_{a} Q(s_{t+n}, a; \theta)." (Section 2.3 The Recurrent Replay Distributed DQN Agent)
- "On the four language tasks in the DMLab suite, we are using the same additional language-LSTM" (Hyper-Parameters)
- "language_select_described_object" (Full Results, Table 3)
- "we cap all (training and evaluation) episodes at 30 minutes (108, 000 environment frames)." (Hyper-Parameters)
- Inference: Classified the task as control/action selection and mapped dimensions/dynamics/attention/state from the presence of image observations, temporal sequences, capped episodes, and an LSTM recurrent state; language input inferred from the "language-LSTM" mention and language-task names.
