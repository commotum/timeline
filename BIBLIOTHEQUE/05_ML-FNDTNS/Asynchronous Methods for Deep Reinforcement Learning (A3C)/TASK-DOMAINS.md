# Asynchronous Methods for Deep Reinforcement Learning (2016)
Source: Asynchronous Methods for Deep Reinforcement Learning (A3C).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (Atari 2600 game playing) | state observations s_t | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Constructed (inferred) | actions a_t | 0D (inferred) | Fixed (inferred) |
| control (TORCS 3D car racing) | RGB image of the current frame | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions a_t | 0D (inferred) | Fixed (inferred) |
| control (continuous motor control: manipulation/locomotion) | physical state (joint positions, velocities, target position); RGB images | 0D (inferred); 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | continuous actions (sampled from a normal distribution) | 0D (inferred) | Fixed (inferred) |
| navigation/control (find rewards in random 3D mazes) | 84 x 84 RGB images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | actions a_t | 0D (inferred) | Fixed (inferred) |

## Summary
The paper applies asynchronous RL to control and navigation tasks across Atari 2600 games, TORCS car racing, MuJoCo continuous control, and a 3D maze environment. Inputs include visual frames (RGB images) and physical state vectors, implying 2D and 0D input dimensions with fixed per-step observation sizes. The tasks use static attention to the provided observation, while state is constructed in settings that use LSTM agents; outputs are per-step actions, including continuous actions in MuJoCo.

## Evidence
### Task: control (Atari 2600 game playing)
- "We perform most of our experiments using the Arcade Learning Environment (Bellemare et al., 2012), which provides a simulator for Atari 2600 games." (Section 5. Experiments)
- "The results show that all four asynchronous methods we presented can successfully train neural network controllers on the Atari domain." (Section 5.1)
- "At each time step t, the agent receives a state s_t and selects an action a_t." (Section 3)
- "a recurrent agent with an additional 256 LSTM cells after the final hidden layer." (Section 5.1)
- Inference: State Dynamic = Constructed because the Atari experiments include an LSTM-based agent ("a recurrent agent with an additional 256 LSTM cells after the final hidden layer," Section 5.1).
- Inference: Attention Dynamic = Static and Out Dimension/Dynamics = 0D/Fixed because the agent receives a state s_t and selects a single action a_t at each time step (Section 3).

### Task: control (TORCS 3D car racing)
- "We also compared the four asynchronous methods on the TORCS 3D car racing game (Wymann et al., 2013)." (Section 5.2)
- "At each step, an agent received only a visual input in the form of an RGB image of the current frame" (Section 5.2)
- "We used the same neural network architecture as the one used in the Atari experiments specified in Supplementary Section 8." (Section 5.2)
- "The network used a convolutional layer with 16 filters of size 8x8 with stride 4" (Section 8)
- "At each time step t, the agent receives a state s_t and selects an action a_t." (Section 3)
- Inference: In Dimension = 2D and In Dynamics = Fixed because the input is an RGB image frame and the architecture uses fixed-size convolutional layers (Sections 5.2 and 8).
- Inference: State Dynamic = Direct because the described architecture is feedforward (convolutional + fully connected) with no recurrent state (Section 8).
- Inference: Attention Dynamic = Static and Out Dimension/Dynamics = 0D/Fixed because the agent maps each observed state to a single action per time step (Section 3).

### Task: control (continuous motor control: manipulation/locomotion)
- "We also examined a set of tasks where the action space is continuous." (Section 5.3)
- "the tasks include many examples of manipulation and locomotion." (Section 5.3)
- "For all the domains we attempted to learn the task using the physical state as input." (Section 9)
- "The physical state consisted of the joint positions and velocities as well as the target position if the task required a target." (Section 9)
- "for three of the tasks (pendulum, pointmass2D, and gripper) we also examined training directly from RGB pixel inputs." (Section 9)
- "the two outputs of the policy network are two real number vectors" (Section 9)
- "the output of the encoder layers were fed to a single layer of 128 LSTM cells." (Section 9)
- Inference: In Dimension = 0D/2D and In Dynamics = Fixed because inputs are physical state vectors and RGB pixels (Section 9).
- Inference: State Dynamic = Constructed because the policy encoder feeds into LSTM cells (Section 9).
- Inference: Attention Dynamic = Static and Out Dimension/Dynamics = 0D/Fixed because the policy outputs a single action vector per time step (Section 9).

### Task: navigation/control (find rewards in random 3D mazes)
- "We performed an additional set of experiments with A3C on a new 3D environment called Labyrinth." (Section 5.4)
- "the agent learning to find rewards in randomly generated mazes." (Section 5.4)
- "We trained an A3C LSTM agent on this task using only 84 x 84 RGB images as input." (Section 5.4)
- "At each time step t, the agent receives a state s_t and selects an action a_t." (Section 3)
- Inference: In Dimension = 2D and In Dynamics = Fixed because the input is fixed-size 84 x 84 RGB images (Section 5.4).
- Inference: State Dynamic = Constructed because the Labyrinth agent is an LSTM agent (Section 5.4).
- Inference: Attention Dynamic = Static and Out Dimension/Dynamics = 0D/Fixed because the agent maps each observed state to a single action per time step (Section 3).
