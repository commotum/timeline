# Some Considerations on Learning to Explore via Meta-Reinforcement Learning (Not specified in the paper)
Source: Some Considerations on Learning to Explore via Meta-Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meta-reinforcement learning control (Krazy World) | Local grid observations (basis-vector tiles) and rollout context \(o, a, r, d\) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Action decisions for agent control | 1D (t) (inferred) | Capped (inferred) |
| Meta-reinforcement learning control (Mazes) | Maze state observations in a 20 x 20 grid over episodes | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Navigation actions to reach goal square | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates meta-reinforcement learning on two control tasks: Krazy World and maze navigation. Both tasks operate on 2D grid observations with temporal interaction, so the supported task domain is spatial control over time. Dynamics are best justified as capped because the paper defines a finite-horizon MDP and bounded grid layouts/observation windows. Attention is dynamic and state is constructed (both inferred) because the policy is optimized to alter sampling behavior and, in RL\(^2\), uses recurrent hidden-state updates for system identification.

## Evidence
### Task: Meta-reinforcement learning control (Krazy World)
- "Results are presented on a new environment we call 'Krazy World': a difficult high-dimensional gridworld..." (Section Abstract)
- "To succeed at Krazy World, a successful meta learning agent will first need to identify and adapt to many different tile types, color palettes, and dynamics." (Section 4.1 Krazy World Environment)
- "In the local mode, the agent only views a 3 x 3 grid centered about itself... We will use local observations." (Appendix A: Krazy-World)
- "Let M=(S,A,P,R,ρ0,γ,T) represent a discrete-time finite-horizon discounted Markov decision process (MDP)." (Section 2 Preliminaries)
- "x_t = [o_{t-1}, a_{t-1}, r_{t-1}, d_{t-1}]" (Section 3.4 E-RL2)
- Inference: `2D (x, y); 1D (t)` input and `1D (t)` output are inferred from local grid observations plus discrete-time interaction/rollouts; `Capped` is inferred from finite horizon \(T\) and fixed local view; `Dynamic` attention is inferred from exploration-focused sampling ("the policy ... defines a sampling process over the state space," Section 3.1); `Constructed` state is inferred from adaptation/update machinery and recurrent hidden-state updates (Section 3.4).

### Task: Meta-reinforcement learning control (Mazes)
- "A collection of maze environments. The agent is placed at a random square within the maze and must learn to navigate the twists and turns to reach the goal square." (Section 4.2 Mazes)
- "The mazes are not rendered, and consequently this task is done with state space only." (Section 4.2 Mazes)
- "The mazes are 20 x 20 squares large." (Section 4.2 Mazes)
- "A good exploratory agent will spend some time learning the maze's layout in a way that minimizes repetition of future mistakes." (Section 4.2 Mazes)
- "RNNs are able to leverage memory, which is more important in mazes than in Krazy World." (Section 4.3 Results)
- Inference: `2D (x, y); 1D (t)` input and `1D (t)` output are inferred from 20x20 spatial states plus sequential navigation; `Capped` is inferred from bounded maze size and finite-horizon MDP framing (Section 2); `Dynamic` attention is inferred from exploratory behavior during runtime interaction; `Constructed` state is inferred from the reported role of RNN memory in maze performance.
