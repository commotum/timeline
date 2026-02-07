# Multi-Game Decision Transformers (Not specified in the paper)
Source: Multi-Game Decision Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (Atari game-playing) | observation images (game frames) + target return token + past actions/rewards (trajectory tokens) | 2D (x, y); 1D (t) | Fixed | Static (inferred) | Direct (inferred) | actions (discrete) | 0D | Fixed |

## Summary
The paper focuses on multi-game Atari control from visual observations, framing trajectories as sequences of image patches, returns, actions, and rewards. Inputs combine spatial game frames and temporal ordering (2D (x, y) plus 1D (t)), while outputs are discrete actions with fixed sizing. The fixed-length context window implies Static attention and a Direct state interface, both inferred from the sequence-model description.

## Evidence
### Task: control (Atari game-playing)
- "trained purely offline can play a suite of up to 46 Atari games simultaneously at close-to-human performance." (Abstract)
- "at every time t receives an observation of the world  $\mathbf{o}^t$ , chooses an action  $a^t$ , and receives a scalar reward  $r^t$ ." (Section 3)
- "$$x = \langle ..., \mathbf{o}_1^t, ..., \mathbf{o}_M^t, \hat{R}^t, a^t, r^t, ... \rangle$$" (Section 3.1)
- "we divide each observation image into a collection of M patches" (Section 3.2)
- "Actions a are already discrete quantities in the environments we consider." (Section 3.2)
- "We set sequence length to 4 game frames for all experiments, which results in sequences of 156 tokens." (Section 4.1)
- Inference: Attention Dynamic is Static (inferred) because the model is a fixed-length decoder-only sequence model with a 4-frame context window; State Dynamic is Direct (inferred) because the policy is defined over the provided sequence without any described external memory or constructed state (Sections 3.1, 4.1).
