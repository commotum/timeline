# Reward-Respecting Subtasks for Model-Based Reinforcement Learning (Not specified in the paper)
Source: Reward-Respecting Subtasks for Model-Based Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main-task reinforcement learning control | State, action, and reward trajectories from episodic interaction | 3D (x, y, z) or (x, y, t) (inferred) | Open (inferred) | Static (inferred) | Constructed | Policy maximizing discounted return and associated value function | 1D (t) (inferred) | Open (inferred) |
| Subtask control via reward-respecting option learning | Off-policy transition tuples and feature vectors for GVF subtasks | 3D (x, y, z) or (x, y, t) (inferred) | Open (inferred) | Static (inferred) | Constructed | Option policy and stopping function for each subtask | 2D (x, y) (inferred) | Fixed (inferred) |
| Option model prediction (reward and transition expectation) | Transition tuples plus learned options | 3D (x, y, z) or (x, y, t) (inferred) | Open (inferred) | Static (inferred) | Constructed | Option reward model r-hat and transition expectation model n-hat | 0D; 2D (x, y) (inferred) | Fixed |
| Model-based planning for value improvement | State-feature vectors and learned action/option models | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed | Updated main-task value function and greedy option/action choices | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper covers a model-based RL pipeline with four distinct task intents: main-task control, reward-respecting subtask control (option learning), option-model prediction, and planning-based value improvement. The experiments are in gridworlds, so the interaction data is spatiotemporal and is classified as 3D (x, y, z) or (x, y, t) (inferred), while planning/value objects are state-indexed 2D structures (inferred). Input dynamics are open for online/off-policy experience streams and fixed for planning-state updates (inferred). Attention is static across the described algorithms (inferred), and state is constructed through learned feature, value, option, and model representations.

## Evidence
### Task: Main-task reinforcement learning control
- "We consider an agent interacting with its environment in a sequence of episodes, each beginning in environment state  $S_0 \doteq s_0 \in \mathbb{S}$  and ending in terminal state  $S_L \doteq \bot^1$  at time step  $L \in \mathbb{N}$ . At time steps t < L, the agent selects an action  $A_t \in \mathcal{A}$ , and in response the environment emits a reward  $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$  and transitions to a next state" (Section 2: Reward-respecting subtasks)
- "The agent's main task is to find a policy  $\pi: \mathcal{S} \times \mathcal{A} \to [0,1]$  that maximizes the expected discounted sum of rewards" (Section 2: Reward-respecting subtasks)
- Inference: `In Dimension = 3D (x, y, z) or (x, y, t)`, `In Dynamics = Open`, and `Out Dimension = 1D (t)` are inferred from time-indexed episodic interaction plus spatial gridworld experiments ("two-room gridworld" and "four-room episodic gridworld"), where policies generate action sequences over variable-length episodes (Section 1; Section 7).

### Task: Subtask control via reward-respecting option learning
- "Given such subtasks, the agent can develop temporally abstract structure for its cognition by following a standard progression in which each subtask is solved to produce an option" (Section 1: The challenge of discovering temporal abstractions)
- "To solve the subtask is to find an option which maximizes (2)." (Section 2: Reward-respecting subtasks)
- "Figure 2: **Option learning experiment**. Algorithm (10) finds an optimal option and its value function for the reward-respecting subtask for attaining the hallway feature." (Figure 2 caption, Section 3)
- Inference: `Attention Dynamic = Static` is inferred because updates consume predefined transition tuples each step (Eq. 10), with no adaptive retrieval mechanism described; `State Dynamic = Constructed` is supported by learned feature/value/policy parameterizations (Sections 2-3).

### Task: Option model prediction (reward and transition expectation)
- "In this section we describe the third step in the STOMP progression: learning a model of the environment's action and option transitions." (Section 4: Model learning)
- "The reward part is a function  $r: S \times O \to \mathbb{R}$  returning the expected cumulative discounted reward if the option were executed starting from the state" (Section 4: Model learning)
- "In this paper we use an *expectation* model (Wan et al., 2019), in which the state-transition part is a function  $\hat{\mathbf{n}}: \mathbb{R}^d \times \mathcal{O} \to \mathbb{R}^d$  such that" (Section 4: Model learning)
- "The procedure described above allows us to efficiently learn both transition and reward models" (Section 4: Model learning)
- Inference: `Out Dimension = 0D; 2D (x, y)` is inferred because the model jointly predicts scalar cumulative reward and a state-feature expectation tied to gridworld state indexing; `In Dynamics = Open` is supported by "50,000 time steps" of off-policy transitions (Section 4).

### Task: Model-based planning for value improvement
- "Our planning method approximates asynchronous value iteration" (Section 5: Planning with options)
- "the planner will iterate over state-feature vectors  $\mathbf{x} \in \mathbb{R}^d$  instead of environmental states" (Section 5: Planning with options)
- "Given options created in the above four ways and the options corresponding to the primitive actions, we computed their models and conducted planning as described earlier in this paper." (Section 7: STOMP in a larger, stochastic gridworld)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from state-indexed gridworld planning/value updates; `In Dynamics = Fixed` and `Out Dynamics = Fixed` are inferred from fixed-size feature vectors and parameter vectors in AVI (Eq. 19); `Attention Dynamic = Static` is inferred because AVI backs up over the available option set per sampled state without a learned retrieval policy (Section 5).
