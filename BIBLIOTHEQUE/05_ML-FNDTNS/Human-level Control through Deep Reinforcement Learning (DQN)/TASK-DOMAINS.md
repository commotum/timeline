# Human-level control through deep reinforcement learning (2015)
Source: Human-level Control through Deep Reinforcement Learning (DQN).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (Atari 2600 game playing) | pixel images (stacked game frames); reward/game score change | 3D (x, y, t) (inferred); 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action (legal game action) | 0D (inferred) | Capped (inferred) |

## Summary
The paper describes a single control task: an agent plays Atari 2600 games by selecting actions to maximize reward from visual observations. Inputs are stacked video frames (plus scalar reward/score changes), yielding fixed-size spatiotemporal observations with static attention, and the state is treated as the observed sequence/stack rather than a separate learned memory. Outputs are discrete game actions from a bounded action set.

## Evidence
### Task: control (Atari 2600 game playing)
- "We consider tasks in which an agent interacts with an environment, in this case the Atari emulator, in a sequence of actions, observations and rewards." (Algorithm, Methods)
- "At each time-step the agent selects an action a_t from the set of legal game actions, A = {1, ..., K}." (Algorithm, Methods)
- "The emulator's internal state is not observed by the agent; instead the agent observes an image x_t" (Algorithm, Methods)
- "In addition it receives a reward r_t representing the change in game score." (Algorithm, Methods)
- "The input to the neural network consists of an 84 × 84 × 4 image produced by the preprocessing map φ." (Model architecture, Methods)
- "applies this preprocessing to the m most recent frames and stacks them to produce the input to the Q-function, in which m=4" (Preprocessing, Methods)
- "The number of valid actions varied between 4 and 18 on the games we considered." (Model architecture, Methods)
- "we can apply standard reinforcement learning methods for MDPs, simply by using the complete sequence s_t as the state representation at time t." (Algorithm, Methods)
- Inference: In/Out dimension, dynamics, attention, and state labels are inferred from fixed stacked-frame input and discrete action set ("84 × 84 × 4 image" — Model architecture; "m most recent frames and stacks them" — Preprocessing; "receives a reward r_t" — Algorithm; "A = {1, ..., K}" and "varied between 4 and 18" — Algorithm/Model architecture; and "complete sequence s_t as the state representation" — Algorithm).

## CSV Output (required)
CSV written to: /home/jake/Developer/timeline/BIBLIOTHEQUE/05_ML-FNDTNS/Human-level Control through Deep Reinforcement Learning (DQN)/.TASK-DOMAINS.csv.tmp.ed9ecded74544f329869f95a5a62bd73
