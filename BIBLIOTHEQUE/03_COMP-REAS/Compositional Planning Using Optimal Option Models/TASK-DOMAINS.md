# Compositional Planning Using Optimal Option Models (2012)
Source: Compositional Planning Using Optimal Option Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (hierarchical manipulation planning) | Tabular MDP states/actions for Tower of Hanoi; transition/reward models; subgoal value models | 0D (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Optimal option models and policy model for goal and subgoals | 0D (inferred) | Fixed (inferred) |
| Control (hierarchical navigation/path planning) | Tabular Nine Rooms gridworld states/actions (N, E, S, W); transition/reward models; doorway subgoal value models | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Optimal option models and policy model for doorways and goal | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper covers control/planning tasks in hierarchical MDPs, instantiated as Tower of Hanoi manipulation planning and Nine Rooms navigation/path planning. Inputs are model-based tabular MDP descriptions (states, actions, transitions, rewards, and subgoal value models), and outputs are composed option models and resulting policy models. The justified dimensional coverage is symbolic/discrete state points for Tower of Hanoi and 2D spatial grids for Nine Rooms (inferred from the explicit grid description). Across tasks, the interface is fixed-size per problem instance, while option selection/termination is state-dependent and the planner constructs higher-level operators.

## Evidence
### Task: Control (hierarchical manipulation planning)
- "We illustrate our framework for compositional planning using two hierarchical MDPs: the Tower of Hanoi problem, and the  $Nine\ Rooms$  problem." (Section 6 Empirical Results)
- "The N-disc Tower of Hanoi problem has a discount factor is  $\gamma = 1$ , each action receives a reward of -1, and episodes terminate upon reaching the goal state (N discs stacked on right peg)." (Section 6 Empirical Results)
- "For the Tower of Hanoi, we use m = 3N + 1 subgoal value models." (Section 6 Empirical Results)
- Inference: `0D`, `Fixed`, `Dynamic`, and `Constructed` are inferred from state-indexed option control and model construction: "A closed-loop policy that is followed for some number of steps, and stops according to a termination condition that also depends on the state, is known as an option" (Section 1 Introduction); "In this paper we have focused on planning with table lookup models" (Section 7 Conclusion); "This is the first MDP planning algorithm to dynamically create its own planning operators." (Section 7 Conclusion).

### Task: Control (hierarchical navigation/path planning)
- "We illustrate our framework for compositional planning using two hierarchical MDPs: the Tower of Hanoi problem, and the  $Nine\ Rooms$  problem." (Section 6 Empirical Results)
- "The level-1 Nine Rooms gridworld is a  $3 \times 3$  grid. The N-level Nine Rooms gridworld contains a  $3 \times 3$  grid of instances of level N-1 problems; neighbouring instances are connected by a width  $3^{N-2}$ doorway; and there is a single goal state in one corner." (Section 6 Empirical Results)
- "We use the primitive actions (moving a disc in Tower of Hanoi; moving N, E, S, W in Nine Rooms) as the base set  $\Omega$ ." (Section 6 Empirical Results)
- Inference: `2D (x, y)`, `Fixed`, `Dynamic`, and `Constructed` are inferred from the explicit grid structure plus state-dependent option control and model construction: "The level-1 Nine Rooms gridworld is a  $3 \times 3$  grid" (Section 6 Empirical Results); "A closed-loop policy that is followed for some number of steps, and stops according to a termination condition that also depends on the state, is known as an option" (Section 1 Introduction); "This is the first MDP planning algorithm to dynamically create its own planning operators." (Section 7 Conclusion).
