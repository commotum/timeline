# Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method (2005)
Source: Neural Fitted Q Iteration (NFQ).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (avoidance; pole balancing) | state (angle, angular velocity) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action/force (left, right, or no force) | 0D (inferred) | Fixed (inferred) |
| control (goal-reaching; mountain car) | state (position, velocity) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action/force (acceleration; -4 or +4) | 0D (inferred) | Fixed (inferred) |
| control (regulator; cartpole regulation) | state (cart position, pole angle, cart velocity, angular velocity) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action/force (-10N or +10N) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates NFQ on three reinforcement-learning control tasks: pole balancing (avoidance), mountain car (goal-reaching), and cartpole regulation (regulator control). Each task uses fixed-size continuous state variables and discrete action choices, so the task inputs/outputs are treated as 0D with Fixed dynamics (inferred). Attention and state dynamics are not discussed explicitly; based on fixed state inputs and action sets, they are labeled Static and Direct (inferred).

## Evidence
### Task: control (avoidance; pole balancing)
- "The task is to balance a pole at the upright position by applying appropriate forces to the system." (Section 5.1 The Pole Balancing Task)
- "The state space is continuous and consists of the angle and the angular velocity." (Section 5.1 The Pole Balancing Task)
- "Three actions are available, left force (-50 N), right force (+50 N) and no force." (Section 5.1 The Pole Balancing Task)
- Inference: In/Out Dimension and Dynamics plus Attention/State are inferred as 0D/Fixed and Static/Direct from the fixed state variables and discrete action set described above. (Section 5.1 The Pole Balancing Task)

### Task: control (goal-reaching; mountain car)
- "The task is to reach the top, which means that then, the position must be larger or equal to 0.7m." (Section 5.2 The Mountain Car Benchmark)
- "initial starting positions are drawn randomly from (-1,0.7), the initial velocity of the car was always set to zero." (Section 5.2 The Mountain Car Benchmark)
- "Two actions are provided to the learning controller, -4 and +4." (Section 5.2 The Mountain Car Benchmark)
- Inference: In/Out Dimension and Dynamics plus Attention/State are inferred as 0D/Fixed and Static/Direct from the fixed position/velocity state variables and discrete action set described above. (Section 5.2 The Mountain Car Benchmark)

### Task: control (regulator; cartpole regulation)
- "The task is to move the cart to a certain position and keep it there while preventing the pole from falling." (Section 5.3 The Cartpole Regulator Benchmark)
- "initial pole angles are randomly drawn from [-0.3, 0.3] (in rad), positions are drawn from [-1., 1.]." (Section 5.3 The Cartpole Regulator Benchmark)
- "cart velocity and angular velocity are initially set to zero." (Section 5.3 The Cartpole Regulator Benchmark)
- "Two actions are available to the learning controller, -10N and +10N." (Section 5.3 The Cartpole Regulator Benchmark)
- Inference: In/Out Dimension and Dynamics plus Attention/State are inferred as 0D/Fixed and Static/Direct from the fixed state variables and discrete action set described above. (Section 5.3 The Cartpole Regulator Benchmark)
