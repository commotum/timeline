# Deterministic Policy Gradient Algorithms (2014)
Source: Deterministic Policy Gradient Algorithms.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continuous bandit optimization (quadratic cost) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | continuous action a | 0D (inferred) | Not specified in the paper. |
| Continuous-action RL benchmark control (mountain car, pendulum, 2D puddle world) | state s (inferred) | 0D (inferred) | Capped | Static (inferred) | Direct (inferred) | continuous action a (inferred) | 0D (inferred) | Capped |
| Octopus arm control to hit a target | 50 continuous state variables (x/y position/velocity of arm nodes; angular position/velocity of base) | 0D (inferred) | Capped | Static (inferred) | Direct (inferred) | 20 action variables controlling arm muscles and base rotation | 0D (inferred) | Capped |

## Summary
The paper applies deterministic policy gradient methods to continuous-action control tasks, including a continuous bandit, standard benchmark control domains (mountain car, pendulum, 2D puddle world), and an octopus arm target-hitting task. Inputs and outputs are described as continuous state/action vectors (explicit for the octopus arm; inferred elsewhere), so the dimensionality is point-like (0D, inferred). The benchmark and octopus tasks are episodic with capped interaction lengths (5000 or 300 steps), while attention and state dynamics are not discussed and are therefore inferred as Static and Direct from the stated policy mapping.

## Evidence
### Task: Continuous bandit optimization (quadratic cost)
- "The problem is a continuous bandit problem with a high-dimensional quadratic cost function,  $-r(a) = (a-a^*)^\top C(a-a^*)$ ." (Section 5.1)
- "We consider action dimensions of m=10,25,50." (Section 5.1)
- Inference: Out Dimension set to 0D (inferred) because the task is defined over an action vector a with specified dimensionality m (point-like action variable). (Section 5.1)

### Task: Continuous-action RL benchmark control (mountain car, pendulum, 2D puddle world)
- "In our second experiment we consider continuous-action variants of standard reinforcement learning benchmarks: mountain car, pendulum and 2D puddle world." (Section 5.2)
- "For all algorithms, episodes were truncated after a maximum of 5000 steps." (Section 5.2)
- "We study reinforcement learning and control problems in which an agent acts in a stochastic environment by sequentially choosing actions over a sequence of time steps, in order to maximise a cumulative reward." (Section 2.1)
- "We model the problem as a Markov decision process (MDP) which comprises: a state space S, an action space A," (Section 2.1)
- "In the remainder of the paper we suppose for simplicity that  $\mathcal{A}=\mathbb{R}^m$  and that  $\mathcal{S}$  is a compact subset of  $\mathbb{R}^d$ ." (Section 2.1)
- Inference: Input/Output set to state s and action a, and In/Out Dimension set to 0D (inferred) based on the MDP definition (state space in $\mathbb{R}^d$, action space in $\mathbb{R}^m$). Attention Dynamic set to Static (inferred) and State Dynamic set to Direct (inferred) because the policy is described as a mapping from state to action with no dynamic attention or constructed state discussed. (Sections 2.1 and 5.2)

### Task: Octopus arm control to hit a target
- "The aim is to learn to control a simulated octopus arm to hit a target." (Section 5.3)
- "There are 50 continuous state variables (x,y position/velocity of the nodes along the upper/lower side of the arm; angular position/velocity of the base) and 20 action variables that control three muscles (dorsal, transversal, central) in each segment as well as the clockwise and counter-clockwise rotation of the base." (Section 5.3)
- "An episode ends when the target is hit (with an additional reward of +50) or after 300 steps." (Section 5.3)
- "We applied the COPDAC-Q algorithm, using a sigmoidal multi-layer perceptron (8 hidden units and sigmoidal output units) to represent the policy  $\mu(s)$ ." (Section 5.3)
- Inference: In/Out Dimension set to 0D (inferred) because the state and action are described as vectors of continuous variables, and Attention Dynamic set to Static (inferred) and State Dynamic set to Direct (inferred) because the policy is expressed as a function of the current state with no dynamic attention or constructed state described. (Section 5.3)
