# Emergent Complexity via Multi-Agent Competition (Not specified in the paper)
Source: Emergent Complexity via Multi-Agent Competition.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Competitive control (Run to Goal) | Observation vector: joint angles, joint velocities, contact forces, opponent relative position and joint angles; Humanoid adds inertia tensor, velocity vector, actuator forces. | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Actions | 1D (t) (inferred) | Capped (inferred) |
| Competitive control (You Shall Not Pass - blocker) | Observation vector: joint angles, joint velocities, contact forces, opponent relative position and joint angles; Humanoid adds inertia tensor, velocity vector, actuator forces. | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Actions | 1D (t) (inferred) | Capped (inferred) |
| Competitive control (You Shall Not Pass - runner) | Observation vector: joint angles, joint velocities, contact forces, opponent relative position and joint angles; Humanoid adds inertia tensor, velocity vector, actuator forces. | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Actions | 1D (t) (inferred) | Capped (inferred) |
| Competitive control (Sumo) | Observation vector: joint angles, joint velocities, contact forces, opponent relative position and joint angles; Humanoid adds inertia tensor, velocity vector, actuator forces; plus torso orientation vector, radial distance from ring edge, time remaining. | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Actions | 1D (t) (inferred) | Capped (inferred) |
| Competitive control (Kick and Defend - kicker) | Observation vector: joint angles, joint velocities, contact forces, opponent relative position and joint angles; Humanoid adds inertia tensor, velocity vector, actuator forces; plus ball relative position, distance to goal, ball position relative to goal posts. | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Actions | 1D (t) (inferred) | Capped (inferred) |
| Competitive control (Kick and Defend - defender) | Observation vector: joint angles, joint velocities, contact forces, opponent relative position and joint angles; Humanoid adds inertia tensor, velocity vector, actuator forces; plus ball relative position, distance to goal, ball position relative to goal posts. | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Actions | 1D (t) (inferred) | Capped (inferred) |
| Stability control (standing under wind perturbations) | Observation vector with opponent parts zeroed. | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Actions | 1D (t) (inferred) | Capped (inferred) |
| Locomotion control (walking) | Not specified in the paper. | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | Actions (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper introduces competitive multi-agent continuous-control tasks in a 3D simulated physics world, spanning race-to-goal, blocking, sumo-style knockdown/out-of-ring, and penalty kick/defend scenarios, plus a transfer task of standing under wind perturbations and a walking baseline. Inputs are fixed observation vectors of body and environment state (with additional task-specific signals for sumo and kick-and-defend), and outputs are continuous control actions. From the episodic, time-stepped training/evaluation descriptions, the tasks operate over temporal sequences with capped episodes (1D (t)), with static observation interfaces; state dynamics differ by policy choice, using direct (MLP) policies for Run-to-Goal/You-Shall-Not-Pass and constructed (LSTM) policies for Sumo/Kick-and-Defend and related transfers.

## Evidence
### Task: Competitive control (Run to Goal)
- "The agent that reaches its goal first wins." (Section 3 Competitive Environments)
- "we use all the joint angles of the agent, its velocity of all its joints" (Section 5.1 Observations)
- "the contact forces acting on the body and the relative position and all the joint angles for the opponent." (Section 5.1 Observations)
- "a set of actions of each agent" (Section 2 Preliminaries, Notation)
- Inference: 1D (t) and capped episode dynamics inferred from time-step/termination framing ("time-step t", "termination time-step", Section 4.1); attention is static from fixed observation vectors ("only observe relevant sub-parts of the state vector", Section 5.1); state is direct from MLP policy use ("MLP policy and value functions for the run-to-goal", Section 5.1).

### Task: Competitive control (You Shall Not Pass - blocker)
- "one agent (the blocker) now has the objective of blocking the other agent from reaching it's goal" (Section 3 Competitive Environments)
- "we use all the joint angles of the agent, its velocity of all its joints" (Section 5.1 Observations)
- "the contact forces acting on the body and the relative position and all the joint angles for the opponent." (Section 5.1 Observations)
- "a set of actions of each agent" (Section 2 Preliminaries, Notation)
- Inference: 1D (t) and capped episode dynamics inferred from time-step/termination framing ("time-step t", "termination time-step", Section 4.1); attention is static from fixed observation vectors ("only observe relevant sub-parts of the state vector", Section 5.1); state is direct from MLP policy use ("MLP policy and value functions for the run-to-goal and you-shall-not-pass", Section 5.1).

### Task: Competitive control (You Shall Not Pass - runner)
- "If the opponent is successful in reaching it's goal then it gets +1000 reward." (Section 3 Competitive Environments)
- "we use all the joint angles of the agent, its velocity of all its joints" (Section 5.1 Observations)
- "the contact forces acting on the body and the relative position and all the joint angles for the opponent." (Section 5.1 Observations)
- "a set of actions of each agent" (Section 2 Preliminaries, Notation)
- Inference: 1D (t) and capped episode dynamics inferred from time-step/termination framing ("time-step t", "termination time-step", Section 4.1); attention is static from fixed observation vectors ("only observe relevant sub-parts of the state vector", Section 5.1); state is direct from MLP policy use ("MLP policy and value functions for the run-to-goal and you-shall-not-pass", Section 5.1).

### Task: Competitive control (Sumo)
- "goal of each agent is to either knock the other agent to the ground" (Section 3 Competitive Environments)
- "we give the torso's orientation vector as the input, the radial distance from the edge of the ring" (Section 5.1 Observations)
- "and the time remaining in the game." (Section 5.1 Observations)
- "a set of actions of each agent" (Section 2 Preliminaries, Notation)
- Inference: 1D (t) and capped episode dynamics inferred from time-step/termination framing ("time-step t", "termination time-step", Section 4.1); attention is static from fixed observation vectors ("only observe relevant sub-parts of the state vector", Section 5.1); state is constructed from LSTM policy use ("LSTM policy and value function for sumo", "single-layer LSTM with 128 hidden state dimension", Section 5.1).

### Task: Competitive control (Kick and Defend - kicker)
- "One agent has to kick a ball through the goal" (Section 3 Competitive Environments)
- "we give the relative position of the ball from the agent, the relative distance of the ball from goal" (Section 5.1 Observations)
- "and the relative position of the ball from the two goal posts." (Section 5.1 Observations)
- "a set of actions of each agent" (Section 2 Preliminaries, Notation)
- Inference: 1D (t) and capped episode dynamics inferred from time-step/termination framing ("time-step t", "termination time-step", Section 4.1); attention is static from fixed observation vectors ("only observe relevant sub-parts of the state vector", Section 5.1); state is constructed from LSTM policy use ("LSTM policy and value function for sumo and kick-and-defend", "single-layer LSTM with 128 hidden state dimension", Section 5.1).

### Task: Competitive control (Kick and Defend - defender)
- "while the other agent defends." (Section 3 Competitive Environments)
- "we give the relative position of the ball from the agent, the relative distance of the ball from goal" (Section 5.1 Observations)
- "and the relative position of the ball from the two goal posts." (Section 5.1 Observations)
- "a set of actions of each agent" (Section 2 Preliminaries, Notation)
- Inference: 1D (t) and capped episode dynamics inferred from time-step/termination framing ("time-step t", "termination time-step", Section 4.1); attention is static from fixed observation vectors ("only observe relevant sub-parts of the state vector", Section 5.1); state is constructed from LSTM policy use ("LSTM policy and value function for sumo and kick-and-defend", "single-layer LSTM with 128 hidden state dimension", Section 5.1).

### Task: Stability control (standing under wind perturbations)
- "faced it with the task of standing while being perturbed by wind forces." (Section 5.2 Learned Behaviors)
- "The agent receives the zero vector for parts of the observation space which correspond to the opponent." (Appendix B.1)
- "a set of actions of each agent" (Section 2 Preliminaries, Notation)
- Inference: 1D (t) and capped episode dynamics inferred from time-step/termination framing ("Episodes last a maximum of 500 time steps", Appendix B.1); attention is static from fixed observation space ("zero vector for parts of the observation space", Appendix B.1); state is constructed from LSTM policy use for sumo ("LSTM policy and value function for sumo", Section 5.1).

### Task: Locomotion control (walking)
- "trained in a single agent environment for the task of walking." (Appendix B.1)
- "Episodes last a maximum of 500 time steps." (Appendix B.1)
- "We used same LSTM policy architecture as used for the Sumo agent" (Appendix B.1)
- Inference: 1D (t) and capped episode dynamics inferred from episodic time-step limit ("Episodes last a maximum of 500 time steps", Appendix B.1); state is constructed from LSTM policy architecture ("same LSTM policy architecture", Appendix B.1); action output inferred from PPO policy use ("using PPO", Appendix B.1).
