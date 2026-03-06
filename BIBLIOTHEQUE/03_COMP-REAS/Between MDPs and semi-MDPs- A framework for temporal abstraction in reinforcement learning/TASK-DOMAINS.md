# Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning (1999)
Source: Between MDPs and semi-MDPs- A framework for temporal abstraction in reinforcement learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (gridworld navigation to goal) | state (grid cell location) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | actions/options (up, down, left, right; hallway options) | 0D (inferred) | Fixed (inferred) |
| control (continuous 2D navigation) | state (continuous two-dimensional location) | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | movement action (0.01 in any direction) | 2D (x, y) (inferred) | Fixed (inferred) |
| control (1D dynamical system to target) | state (position and velocity) | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | action (applied force a_t) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper develops reinforcement learning control with options and illustrates it on a discrete gridworld navigation task, a continuous 2D navigation task with landmark controllers, and a 1D dynamical control task moving a mass to a target. Inputs span 2D spatial grids and continuous position/velocity states, while outputs are discrete movement actions or continuous control actions; only the gridworld example implies a fixed finite state/action structure, while bounds for continuous tasks are not specified. Attention and state dynamics are not explicitly labeled; the interaction model implies static attention to the current state and constructed state via temporally extended options (inferred).

## Evidence
### Task: control (gridworld navigation to goal)
- "As a simple illustration of planning with options, consider the rooms example, a gridworld environment of four rooms as shown in Fig. 2." (Section 3)
- "The cells of the grid correspond to the states of the environment." (Section 3)
- "From any state the agent can perform one of four actions, up, down, left or right, which have a stochastic effect." (Section 3)
- "Now consider a sequence of planning tasks for navigating within the grid to a designated goal state" (Section 3)
- Inference: Classified input as `2D (x, y)` and fixed because "The cells of the grid correspond to the states of the environment."; output as `0D` and fixed because "From any state the agent can perform one of four actions, up, down, left or right"; attention as static because "On each time step, t, the agent perceives the state of the environment,  $s_t \\in \\mathcal{S}$ , and on that basis chooses a primitive action,  $a_t \\in \\mathcal{A}_{s_t}$ ." (Section 1); state as constructed because "Options consist of three components: a policy  $\\pi: \\mathcal{S} \\times \\mathcal{A} \\to [0, 1]$ , a termination condition  $\\beta: \\mathcal{S}^+ \\to [0, 1]$ , and an initiation set  $\\mathcal{I} \\subseteq \\mathcal{S}$ ." (Section 2).

### Task: control (continuous 2D navigation)
- "Here the task is to navigate from a start location to a goal location within a continuous two-dimensional state space." (Section 4)
- "The actions are movements of 0.01 in any direction from the current state." (Section 4)
- "The task (top) is to navigate from S to G in minimum time using options based on controllers that run each to one of seven landmarks" (Fig. 7 caption)
- Inference: Marked output as `2D (x, y)` and fixed because actions are "movements of 0.01 in any direction"; attention as static because "On each time step, t, the agent perceives the state of the environment,  $s_t \\in \\mathcal{S}$ , and on that basis chooses a primitive action,  $a_t \\in \\mathcal{A}_{s_t}$ ." (Section 1); state as constructed because the task uses "options based on controllers that run each to one of seven landmarks" (Fig. 7 caption).

### Task: control (1D dynamical system to target)
- "Fig. 8 shows results for an example using controllers/options with dynamics. The task here is to move a mass along one dimension from rest at position 0 to rest at position 2," (Section 4)
- "The system is a mass moving in one dimension:  $x_{t+1} = x_t + \dot{x}_{t+1}$ ,  $\dot{x}_{t+1} = \dot{x}_t + a_t - 0.175\dot{x}_t$  where  $x_t$  is the position,  $\dot{x}_t$  the velocity, 0.175 a coefficient of friction, and the action  $a_t$  an applied force." (Fig. 8 caption)
- "Two controllers are provided as options, one that drives the position to zero velocity at  $x^* = 1$  and the other to  $x^* = 2$ ." (Fig. 8 caption)
- Inference: Treated the input as `2D (x, y)` because the state includes position and velocity; output as `0D` and fixed because the action is a scalar applied force; attention as static because "On each time step, t, the agent perceives the state of the environment,  $s_t \\in \\mathcal{S}$ , and on that basis chooses a primitive action,  $a_t \\in \\mathcal{A}_{s_t}$ ." (Section 1); state as constructed because "Two controllers are provided as options" (Fig. 8 caption).
