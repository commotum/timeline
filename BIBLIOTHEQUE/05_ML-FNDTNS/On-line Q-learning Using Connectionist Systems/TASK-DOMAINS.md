# On-line Q-learning Using Connectionist Systems (1994)
Source: On-line Q-learning Using Connectionist Systems.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (action-value) | state vector (sensor readings, goal distance/angle) and action (discrete action) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | scalar Q-value / predicted return | 0D (inferred) | Fixed (inferred) |
| control (robot navigation) | sensor readings (range finders) plus goal distance/angle | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | discrete movement actions (turn/move) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates connectionist Q-learning on a robot navigation control task where a simulated robot uses range-finder and goal-relative sensors to reach a goal while avoiding obstacles. The learning system also explicitly predicts action-value returns (Q-values) for each state-action pair using neural networks with fixed-size inputs and scalar outputs. Task interactions are sequential over time with trials limited to a maximum number of steps, implying capped temporal dynamics for both observations and actions. Inputs are fixed sensor vectors (static attention), and the system maintains learned Q-function representations via neural networks, indicating constructed state.

## Evidence
### Task: prediction (action-value)
- "learn Q-function, which is a prediction of the return associated with each action  $a \in A$  in each state." (Section 2.1 Q-Learning)
- "use of back-propagation neural networks to store the information learnt by the Q-learning algorithm" (Section 1 Introduction)
- "The Q-function was represented by 6 neural networks, one for each available action." (Section 5.2 Experimental Details)
- "Each network had 26 inputs, 3 hidden nodes, and a single output," (Section 5.2 Experimental Details)
- Inference: Input/Output dimensions and dynamics labeled 0D/Fixed, and attention marked Static, because the Q-function is implemented with a fixed-size input vector and a single scalar output per action; state marked Constructed because the system stores learned information in neural network parameters (Sections 1, 5.2).

### Task: control (robot navigation)
- "a simulated mobile robot is trained to guide itself to a goal position in the presence of obstacles." (Abstract)
- "The robot is simulated with five range finding inputs which give it accurate distance measurements to obstructions," (Section 5.1 The Robot Environment)
- "It also always knows the distance and angle to the goal relative to its current position and facing." (Section 5.1 The Robot Environment)
- "The simulated robot was trained with 6 actions available to it: turn left 15°, turn right 15°, or keep the same heading," (Section 5.2 Experimental Details)
- "and either move forward a fixed distance d, or remain on the same spot." (Section 5.2 Experimental Details)
- "the robot is allowed only a limited number of steps in which to reach the goal." (Section 5.1 The Robot Environment)
- Inference: Input/Output treated as time-ordered sequences with capped dynamics because actions are taken at each step until a limited maximum number of steps; attention marked Static from the fixed sensor suite; state marked Constructed from the learned Q-function used to guide control (Sections 5.1, 5.2).
