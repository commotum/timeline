# Addressing Function Approximation Error in Actor-Critic Methods (2018)
Source: Twin Delayed DDPG (TD3).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continuous control | Environment states / state trajectories | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Continuous actions / action trajectories | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper targets continuous control in actor-critic reinforcement learning across MuJoCo/OpenAI Gym domains. The model interaction is temporal, with states and actions occurring at discrete time steps, which supports a 1D (t) input/output characterization. The reported setup uses bounded rollout/update horizons, so dynamics are classified as Capped. Attention and state handling are inferred as Static and Direct from the described feedforward policy mapping from current state to action.

## Evidence
### Task: Continuous control
- "In reinforcement learning problems with discrete action spaces, the issue of value overestimation as a result of function approximation errors is well-studied. However, similar issues with actor-critic methods in continuous control domains have been largely left untouched." (Section 1. Introduction)
- "At each discrete time step t, with a given state  $s \in \mathcal{S}$ , the agent selects actions  $a \in \mathcal{A}$  with respect to its policy  $\pi: \mathcal{S} \to \mathcal{A}$ , receiving a reward r and the new state of the environment s'." (Section 3. Background)
- "To evaluate our algorithm, we measure its performance on the suite of MuJoCo continuous control tasks (Todorov et al., 2012), interfaced through OpenAI Gym (Brockman et al., 2016) (Figure 4)." (Section 6.1. Evaluation)
- Inference: In Dimension/Out Dimension are set to "1D (t) (inferred)" because the task is explicitly indexed by discrete time steps ("At each discrete time step t") and return is accumulated along trajectories. In Dynamics/Out Dynamics are set to "Capped (inferred)" because Algorithm 1 specifies "for t=1 to T do" and Supplementary D references episodes running until a "max horizon." Attention Dynamic and State Dynamic are set to "Static (inferred)" and "Direct (inferred)" based on fixed feedforward actor-critic mappings (Section 6.1 architecture description) and direct policy mapping "$\pi: \mathcal{S} \to \mathcal{A}$".
