# Solving the Rubik's Cube Without Human Knowledge (Not specified in the paper)
Source: Solving the Rubik's Cube Without Human Knowledge.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rubik's Cube solving (control) | Rubik's Cube state (scrambled configuration) | 3D (x, y, z) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Move sequence to solved state | 1D (t) (inferred) | Capped (inferred) |
| State-value prediction | Rubik's Cube state representation (20x24) | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | State value estimate (scalar v) | 0D (inferred) | Fixed (inferred) |
| Policy prediction (move-probability estimation) | Rubik's Cube state representation (20x24) | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Move probability vector p over 12 actions | 0D (inferred) | Fixed (inferred) |

## Summary
The paper’s primary task is sequential control for Rubik’s Cube solving, implemented with MCTS guided by a learned network. The same model explicitly handles two prediction tasks: state-value estimation and policy (move-probability) estimation from cube states. Inputs are described both as a 3D cube domain and as a fixed 20x24 state representation for the network, while outputs span scalar/value estimates, fixed action distributions, and variable-length action sequences. The solver behavior supports Dynamic attention and Constructed state through runtime tree expansion and per-node search memory.

## Evidence
### Task: Rubik's Cube solving (control)
- "Our algorithm is able to solve 100% of randomly scrambled cubes while achieving a median solve length of 30 moves" (Abstract)
- "We employ an asynchronous Monte Carlo Tree Search augmented with our trained neural network  $f_{\theta}$  to solve the cube from a given starting state  $s_0$ ." (Section 4.2 Solver)
- "If  $s_{\tau}$  is the solved state, then the tree T of the simulation is extracted and converted into an undirected graph with unit weights. A full breath-first search is then applied on T to find the shortest predicted path from the starting state to solution." (Section 4.2 Solver)
- Inference: In Dynamics is Fixed from the fixed cube structure/state per step ("The 3x3x3 Rubik's cube is a classic 3-Dimensional combination puzzle."), Attention is Dynamic from runtime tree-policy action selection ("an action is selected by choosing,  $A_t = \operatorname{argmax}_a U_{s_t}(a) + Q_{s_t}(a)$"), State is Constructed from explicit per-node memories ("Each state,  $s \in T$ , has a memory attached to it storing:  $N_s(a)$ , the number of times an action a has been taken from state s,  $W_s(a)$ , the maximal value of action a from state s,  $L_s(a)$ , the current virtual loss for action a from state s, and  $P_s(a)$ , the prior probability of action a from state s."), and Out Dynamics is Capped with sequence output under bounded search ("The simulation is performed until either  $s_{\tau}$  is the solved state or the simulation exceeds a fixed maximum computation time.") (Sections 3 and 4.2)

### Task: State-value prediction
- "ADI is an iterative supervised learning procedure which trains a deep neural network  $f_{\theta}(s)$  with parameters  $\theta$  which takes an input state s and outputs a value and policy pair (v, p)." (Section 4.1 Autodidactic Iteration)
- "The outputs of the network are a 1 dimensional scalar v, representing the value" (Section 5 Results)
- "This results in a 20x24 state representation" (Section 3 The Rubik's Cube)
- Inference: Attention is Static and State is Direct because the predictor is a feed-forward mapping over fixed state input without described persistent memory ("We used a feed forward network as the architecture for  $f_{\theta}$  as shown in Figure 4." and "Each layer is fully connected.") (Section 5 Results and Figure 4)

### Task: Policy prediction (move-probability estimation)
- "The policy output p is a vector containing the move probabilities for each of the 12 possible moves from that state." (Section 4.1 Autodidactic Iteration)
- "The outputs of the network are a 1 dimensional scalar v, representing the value, and a 12 dimensional vector p, representing the probability of selecting each of the possible moves." (Section 5 Results)
- "At each timestep, t, the agent observes a state  $s_t \in \mathcal S$  and takes an action  $a_t \in \mathcal A$  with  $\mathcal A := \{F, F', \ldots, D, D'\}$ ." (Section 3 The Rubik's Cube)
- Inference: In/Out Dynamics are Fixed from fixed-size state/action interfaces (20x24 input; 12-move output), and Attention is Static with Direct state for the feed-forward prediction head (Sections 3, 4.1, and 5)
