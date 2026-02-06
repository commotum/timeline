# Dota 2 with Large Scale Deep Reinforcement Learning (2021)
Source: Dota 2 with Large Scale Deep Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (Dota 2 gameplay) (inferred) | History of game observations (semantic game state) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Discrete actions (movement, attack, etc.) | 1D (t) (inferred) | Capped (inferred) |
| Value prediction (state-value function) (inferred) | History of game observations (semantic game state) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Value function estimate | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper describes an RL system for playing Dota 2 that maps a history of semantic game observations to discrete actions, and also outputs a value estimate from the same recurrent network. The tasks are temporal in nature (observation/action streams over time), and the interaction occurs over long but finite episodes (approximately 20,000 steps), indicating capped dynamics. Attention is static and state is constructed via an LSTM-based policy, inferred from the fixed observation interface and recurrent architecture.

## Evidence
### Task: Control (Dota 2 gameplay) (inferred)
- "Each timestep, OpenAI Five receives an *observation* from the game engine" (Section 3.1 Playing Dota using AI)
- "OpenAI Five then returns a discrete *action* to the game engine, encoding a desired movement, attack, etc." (Section 3.1 Playing Dota using AI)
- "We define a policy (π) as a function from the history of observations to a probability distribution over actions" (Section 3.1 Playing Dota using AI)
- "The neural network consists primarily of a single-layer 4096-unit LSTM" (Section 3.1 Playing Dota using AI)
- "OpenAI Five selects an action every fourth frame, yielding approximately 20,000 steps per episode." (Section 2 Dota 2)
- Inference: Labeled the task as control and set 1D (t) input/output because the policy operates over a history of timestep observations and actions; marked dynamics as capped based on the ~20,000-step episodes; marked attention as static because observations are provided each timestep without a described selection mechanism; marked state as constructed due to the recurrent (LSTM) policy. (Supported by the quotes above and the Figure 1 caption.)

### Task: Value prediction (state-value function) (inferred)
- "The LSTM state is projected to obtain the policy outputs (actions and value function)." (Figure 1)
- "In addition to the action logits, the value function is computed as another linear projection of the LSTM state." (Section H Neural Network Architecture)
- "We define a policy (π) as a function from the history of observations to a probability distribution over actions" (Section 3.1 Playing Dota using AI)
- "The neural network consists primarily of a single-layer 4096-unit LSTM" (Section 3.1 Playing Dota using AI)
- "OpenAI Five selects an action every fourth frame, yielding approximately 20,000 steps per episode." (Section 2 Dota 2)
- Inference: Treated this as a value-prediction task over a temporal observation stream; set 1D (t) input/output and capped dynamics by the episodic timestep structure; marked attention as static and state as constructed based on the fixed observation interface and LSTM recurrence. (Supported by the quotes above.)
