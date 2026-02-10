# RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning (2016)
Source: RL$^2$- Fast Reinforcement Learning via Slow Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-armed bandit control | Bandit interaction history (arm actions, rewards, termination flags; stateless environment with placeholder state) | 1D (t) | Capped | Dynamic (inferred) | Constructed | Arm-selection actions | 1D (t) | Capped |
| Tabular MDP control | State/action/reward/termination interaction trajectories in finite tabular MDPs | 1D (t) | Capped | Dynamic (inferred) | Constructed | Discrete control actions | 1D (t) | Capped |
| Vision-based navigation control | RGB image observations with action/reward/termination history across episodes in maze trials | 3D (x, y, t) (inferred) | Capped | Dynamic (inferred) | Constructed | Navigation actions | 1D (t) | Capped |

## Summary
The paper evaluates RL^2 on three reinforcement-learning control tasks: multi-armed bandits, tabular MDPs, and vision-based maze navigation. The task interface is temporal in all cases (1D (t)), and the visual task adds spatial image structure, which supports a 3D (x, y, t) input classification (inferred). Episode counts and horizons are explicitly bounded in the experiments, so both input and output dynamics are Capped. The policy maintains recurrent hidden activations across episodes (Constructed state), and its continual exploration/exploitation adaptation supports Dynamic attention as an inference.

## Evidence
### Task: Multi-armed bandit control
- "Multi-armed bandit problems are a subset of MDPs where the agent's environment is stateless." (Section 3.1 Multi-armed bandits)
- "Specifically, there are k arms (actions), and at every time step, the agent pulls one of the arms, say i, and receives a reward drawn from an unknown distribution: our experiments take each arm to be a Bernoulli distribution with parameter  $p_i$ ." (Section 3.1 Multi-armed bandits)
- "The goal is to maximize the total reward obtained over a fixed number of time steps." (Section 3.1 Multi-armed bandits)
- "Each timestep, it receives the tuple (s,a,r,d) as input, which is embedded using a function  $\phi(s,a,r,d)$  and provided as input to an RNN." (Section 2.3 Policy Representation)
- "The output of the GRU is fed to a fully connected layer followed by a softmax function, which forms the distribution over actions." (Section 2.3 Policy Representation)
- "At the end of an episode, the hidden state of the policy is preserved to the next episode, but not preserved between trials." (Section 2.2 Formulation)
- Inference: Attention Dynamic is labeled Dynamic (inferred) because the paper states "the agent is forced to integrate all the information it has received, including past actions, rewards, and termination flags, and adapt its strategy continually." (Section 2.2 Formulation)

### Task: Tabular MDP control
- "Hence, we perform further experiments using randomly generated tabular MDPs, where there is a finite number of possible states and actions—small enough that the transition probability distribution can be explicitly given as a table." (Section 3.2 Tabular MDPs)
- "The distribution over MDPs is constructed with  $|\mathcal{S}|=10$ ,  $|\mathcal{A}|=5$ ." (Section 3.2 Tabular MDPs)
- "We set the horizon for each episode to be T=10, and an episode always starts on the first state." (Section 3.2 Tabular MDPs)
- "Each timestep, it receives the tuple (s,a,r,d) as input, which is embedded using a function  $\phi(s,a,r,d)$  and provided as input to an RNN." (Section 2.3 Policy Representation)
- "The output of the GRU is fed to a fully connected layer followed by a softmax function, which forms the distribution over actions." (Section 2.3 Policy Representation)
- "At the end of an episode, the hidden state of the policy is preserved to the next episode, but not preserved between trials." (Section 2.2 Formulation)
- Inference: Attention Dynamic is labeled Dynamic (inferred) because the policy must condition behavior on belief and accumulated history: "the agent must act differently according to its belief over which MDP it is currently in." (Section 2.2 Formulation)

### Task: Vision-based navigation control
- "For the second question, we evaluate  $RL^2$  on a vision-based navigation task." (Section 3 Evaluation)
- "agent is asked to navigate a randomly generated maze to find a randomly placed target<sup>2</sup>." (Section 3.3 Visual Navigation)
- "It can interact with the maze for multiple episodes, during which the maze structure and target position are held fixed." (Section 3.3 Visual Navigation)
- "We use a simple training setup, where we use small mazes of size  $5\times 5$ , with 2 episodes of interaction, each with horizon up to 250." (Section 3.3 Visual Navigation)
- "In addition, we also study its extrapolation behavior along two axes, by (1) testing on large mazes of size  $9\times 9$  (see Figure 4c) and (2) running the agent for up to 5 episodes in both small and large mazes." (Section 3.3 Visual Navigation)
- "We rescale the images to have width 40 and height 30 with RGB channels preserved, and we recenter the RGB values to lie within range [-1,1]." (Section A.3 Visual Navigation)
- "At the end of an episode, the hidden state of the policy is preserved to the next episode, but not preserved between trials." (Section 2.2 Formulation)
- Inference: In Dimension is labeled 3D (x, y, t) (inferred) because the task uses 2D RGB observations over temporal interaction episodes; Attention Dynamic is labeled Dynamic (inferred) because "The optimal strategy is to explore the maze efficiently during the first episode, and after locating the target, act optimally against the current maze and target based on the collected information." (Section 3.3 Visual Navigation)
