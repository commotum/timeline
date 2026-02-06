# Agent57: Outperforming the Atari Human Benchmark (2020)
Source: Agent57- Outperforming the Atari Human Benchmark.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (Atari 2600 game playing) | current observation (Atari frames) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | action | 0D (inferred) | Not specified in the paper. |
| Control (gridworld "random coin") | current observation (gridworld) | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | action (up, down, left right) | 0D (inferred) | Fixed |

## Summary
Agent57 is evaluated as a reinforcement-learning control system on Atari 2600 games and on a minimalistic 15x15 gridworld ("random coin"). Inputs are current observations (Atari frames or gridworld observations) and outputs are discrete actions. The paper supports 2D spatial inputs with fixed size for the gridworld and (inferred) fixed-size frames for Atari; outputs are single actions (0D). Attention is not discussed and is marked Static (inferred) based on fixed observation input, while the use of a recurrent network implies Constructed state (inferred).

## Evidence
### Task: Control (Atari 2600 game playing)
- "We propose Agent57, the first deep RL agent that outperforms the standard human benchmark on all 57 Atari games." (Abstract)
- "x is the current observation, a is an action" (Section 2: Background: Never Give Up (NGU))
- "including the standard preprocessing of Atari frames." (Section 4: Experiments)
- "NGU trains a recurrent neural network  $Q(x,a,j;\theta)$" (Section 2: Background: Never Give Up (NGU))
- "parameters of the network (including the recurrent state)." (Section 2: Background: Never Give Up (NGU))
- Inference: Input dimension and input dynamics are marked 2D (x, y) and Fixed based on the use of Atari frames with standard preprocessing; Attention is marked Static because the model consumes the current observation without any described runtime selection; State is marked Constructed because the model is a recurrent neural network with a recurrent state; Output dimension is marked 0D because the output is a single action. (Supported by quotes above.)

### Task: Control (gridworld "random coin")
- "a minimalistic gridworld environment, called \"random coin\"." (Section 4.2: State-Action Value Function Parameterization)
- "It consists of an empty room of size  $15 \times 15$  where a coin and an agent are randomly placed." (Section 4.2: State-Action Value Function Parameterization)
- "The agent can take four possible actions (up, down, left right)" (Section 4.2: State-Action Value Function Parameterization)
- "x is the current observation, a is an action" (Section 2: Background: Never Give Up (NGU))
- "parameters of the network (including the recurrent state)." (Section 2: Background: Never Give Up (NGU))
- Inference: Attention is marked Static because the model takes the current observation as input with no runtime selection described; State is marked Constructed because the model uses a recurrent neural network with a recurrent state; Output dimension is marked 0D because the output is a single discrete action. (Supported by quotes above.)
