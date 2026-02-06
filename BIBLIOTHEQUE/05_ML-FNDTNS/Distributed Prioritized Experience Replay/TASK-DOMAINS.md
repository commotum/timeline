# DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY (Not specified in the paper.)
Source: Distributed Prioritized Experience Replay.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Atari game playing (Ape-X DQN) | game frames (pixels) | 3D (x, y, z) or (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| Manipulator: bring a ball to a specified location | observation/state features | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | continuous action vector | 1D (t) (inferred) | Fixed (inferred) |
| Humanoid standing | observation/state features | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | continuous action vector | 1D (t) (inferred) | Fixed (inferred) |
| Humanoid walking | observation/state features | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | continuous action vector | 1D (t) (inferred) | Fixed (inferred) |
| Humanoid running | observation/state features | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | continuous action vector | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper covers Atari game playing from pixel frames and four continuous control tasks in MuJoCo (manipulator ball placement and humanoid standing, walking, running) using feature observations. Inputs are either stacked game frames (3D spatiotemporal) or fixed-length observation vectors (1D), and outputs are single actions or continuous action vectors; input/output dynamics are inferred as fixed from preprocessing and stated action/observation dimensionalities. Attention and state dynamics are not explicitly described and are inferred as static/direct from policies that map the current state/observation to an action.

## Evidence
### Task: Atari game playing (Ape-X DQN)
- "In our first set of experiments we evaluate Ape-X DQN on Atari, and show state of the art results on this standard reinforcement learning benchmark." (Section 4.1 Atari)
- "The frames received from the environment are preprocessed on the actor side with the standard transformations introduced by DQN. This includes greyscaling, frame stacking, repeating actions 4 times, and clipping rewards to [-1, 1]." (Section C Atari: Additional Details)
- "the actors interact with their own instances of the environment by selecting actions according to a shared neural network" (Abstract)
- "a_{t-1} \leftarrow \pi_{\theta_{t-1}}(s_{t-1})" (Algorithm 1 Actor)
- Inference: In Dimension 3D (x, y, z) or (x, y, t) and In Dynamics Fixed inferred from the use of stacked game frames; Out Dimension 0D and Out Dynamics Fixed inferred from per-step action selection; Attention Static and State Direct inferred from the policy acting directly on the current state. (Section C Atari: Additional Details; Abstract; Algorithm 1 Actor)

### Task: Manipulator: bring a ball to a specified location
- "In the manipulator domain the agent must learn to bring a ball to a specified location." (Section 4.2 Continuous Control)
- "Manipulator is a 2-dimensional planar arm with  $|\mathcal{A}|=2$ ,  $|\mathcal{S}|=22$  and  $|\mathcal{O}|=37$ , which receives reward for catching a randomly-initialized moving ball." (Section D Continuous Control: Additional Details)
- "The policy network outputs an action  $A_t = \pi(S_t, \phi) \in \mathbb{R}^m$ ." (Section 3.2 APE-X DPG)
- Inference: In/Out Dimension 1D (t) and In/Out Dynamics Fixed inferred from stated action/observation dimensionalities; Attention Static and State Direct inferred from the policy mapping the current state to an action without any described dynamic selection or constructed state. (Section D Continuous Control: Additional Details; Section 3.2 APE-X DPG)

### Task: Humanoid standing
- "In the humanoid domain the agent must learn to control a humanoid body to solve three distinct tasks of increasing complexity: Standing, Walking and Running." (Section 4.2 Continuous Control)
- "Humanoid is a humanoid walker with action, state and observation dimensionalities  $|\mathcal{A}|=21$ ,  $|\mathcal{S}|=55$  and  $|\mathcal{O}|=67$  respectively." (Section D Continuous Control: Additional Details)
- "The policy network outputs an action  $A_t = \pi(S_t, \phi) \in \mathbb{R}^m$ ." (Section 3.2 APE-X DPG)
- Inference: In/Out Dimension 1D (t) and In/Out Dynamics Fixed inferred from stated action/observation dimensionalities; Attention Static and State Direct inferred from the policy mapping the current state to an action without any described dynamic selection or constructed state. (Section D Continuous Control: Additional Details; Section 3.2 APE-X DPG)

### Task: Humanoid walking
- "In the humanoid domain the agent must learn to control a humanoid body to solve three distinct tasks of increasing complexity: Standing, Walking and Running." (Section 4.2 Continuous Control)
- "Humanoid is a humanoid walker with action, state and observation dimensionalities  $|\mathcal{A}|=21$ ,  $|\mathcal{S}|=55$  and  $|\mathcal{O}|=67$  respectively." (Section D Continuous Control: Additional Details)
- "The policy network outputs an action  $A_t = \pi(S_t, \phi) \in \mathbb{R}^m$ ." (Section 3.2 APE-X DPG)
- Inference: In/Out Dimension 1D (t) and In/Out Dynamics Fixed inferred from stated action/observation dimensionalities; Attention Static and State Direct inferred from the policy mapping the current state to an action without any described dynamic selection or constructed state. (Section D Continuous Control: Additional Details; Section 3.2 APE-X DPG)

### Task: Humanoid running
- "In the humanoid domain the agent must learn to control a humanoid body to solve three distinct tasks of increasing complexity: Standing, Walking and Running." (Section 4.2 Continuous Control)
- "Humanoid is a humanoid walker with action, state and observation dimensionalities  $|\mathcal{A}|=21$ ,  $|\mathcal{S}|=55$  and  $|\mathcal{O}|=67$  respectively." (Section D Continuous Control: Additional Details)
- "The policy network outputs an action  $A_t = \pi(S_t, \phi) \in \mathbb{R}^m$ ." (Section 3.2 APE-X DPG)
- Inference: In/Out Dimension 1D (t) and In/Out Dynamics Fixed inferred from stated action/observation dimensionalities; Attention Static and State Direct inferred from the policy mapping the current state to an action without any described dynamic selection or constructed state. (Section D Continuous Control: Additional Details; Section 3.2 APE-X DPG)
