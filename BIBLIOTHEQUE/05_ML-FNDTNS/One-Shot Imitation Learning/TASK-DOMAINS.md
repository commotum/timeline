# One-Shot Imitation Learning (Not specified in the paper)
Source: One-Shot Imitation Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control/manipulation (block stacking) | Demonstration trajectory + current observation (block positions, gripper state) | 1D (t); 3D (x, y, z) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Actions/controls (robot arm) | 1D (t) | Open (inferred) |
| Control (particle reaching) | Demonstration trajectory + current state (agent 2D location, landmark 2D locations) | 1D (t); 2D (x, y) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Actions/controls (2D force) | 1D (t) | Open (inferred) |

## Summary
The paper studies one-shot imitation control for two task families: robotic block stacking with 3D object positions and particle reaching with 2D landmark positions, both conditioned on demonstration trajectories plus current state. Outputs are action sequences for control. The input size varies with time and number of objects/landmarks, and the model uses attention and learned embeddings, supporting dynamic attention and constructed state (inferred from the described attention and embedding mechanisms).

## Evidence
### Task: Control/manipulation (block stacking)
- "goal is to control a 7-DOF Fetch robotic arm to stack various numbers of cube-shaped blocks into a specific configuration specified by the user." (Section 3.2 Block Stacking Tasks)
- "In a typical task, an observation is a list of (x,y,z) object positions relative to the gripper, and information if gripper is opened or closed." (Section 3.2 Block Stacking Tasks)
- "The number of objects may vary across different task instances." (Section 3.2 Block Stacking Tasks)
- "Our learned policy takes as input: (i) the current observation, and (ii) one demonstration" (Section 1 Introduction)
- "The policy outputs the current controls." (Section 1 Introduction)
- "Our approach heavily relies on an attention model over the demonstration and an attention model over the current observation." (Section 2 Related Work)
- "The demonstration network receives a demonstration trajectory as input, and produces an embedding of the demonstration to be used by the policy." (Section 4.1 Demonstration Network)
- "consume a very long demonstration sequence and, effectively, emit a long sequence of actions." (Section 2 Related Work)
- "demonstrations can span hundreds to thousands of time steps" (Section 4.1 Demonstration Network)
- Inference: In Dynamics and Out Dynamics are Open, and Attention/State are Dynamic/Constructed because demonstrations and object counts vary and the model uses attention and learned embeddings. (Based on the quotes above from Sections 3.2, 2, and 4.1.)

### Task: Control (particle reaching)
- "The particle reaching problem is a very simple family of tasks." (Appendix A: Illustrative Example: Particle Reaching)
- "we control a point robot to reach a specific landmark" (Appendix A: Illustrative Example: Particle Reaching)
- "The agent receives its own 2D location, as well as the 2D locations of each of the landmarks." (Appendix A: Illustrative Example: Particle Reaching)
- "Without a demonstration, the robot does not know which landmark it should reach" (Appendix A: Illustrative Example: Particle Reaching)
- "The robot is a point mass controlled with 2-dimensional force." (Appendix A, Figure 1)
- "the LSTM outputs a weighting over the different landmarks from the demonstration sequence." (Appendix A: Illustrative Example: Particle Reaching)
- "Our learned policy takes as input: (i) the current observation, and (ii) one demonstration" (Section 1 Introduction)
- "consume a very long demonstration sequence and, effectively, emit a long sequence of actions." (Section 2 Related Work)
- "The demonstration network receives a demonstration trajectory as input, and produces an embedding of the demonstration to be used by the policy." (Section 4.1 Demonstration Network)
- Inference: In Dynamics and Out Dynamics are Open, and Attention/State are Dynamic/Constructed because the task uses variable landmark sets and demonstrations, and attention/embedding mechanisms are described. (Based on the quotes above from Appendix A and Sections 2 and 4.1.)
