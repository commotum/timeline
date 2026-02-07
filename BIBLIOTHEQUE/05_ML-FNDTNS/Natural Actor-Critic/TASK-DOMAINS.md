# Natural Actor-Critic (2008)
Source: Natural Actor-Critic.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (continuous-state/action MDP policy optimization) | state $x_t$ in $\mathbb{X}$ | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | action $u_t$ in $\mathbb{U}$ | 1D (t) (inferred) | Open (inferred) |
| Cart-Pole Balancing | cart-pole state $\mathbf{x} = [\mathbf{x}, \dot{\mathbf{x}}, \theta, \dot{\theta}]^{\mathrm{T}}$ | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | action $\mathbf{u} = F$ | 1D (t) (inferred) | Open (inferred) |
| Point-to-point motor primitive learning | movement plans $(\mathbf{q}_d, \dot{\mathbf{q}}_d)$ for DOF robot systems | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | desired joint velocity $\dot{q}_{d,k}$ / movement trajectory (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Baseball swing (hitting a T-ball) | motor primitive planning for a seven DOF robot task (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | robot swing/batting trajectory (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper defines a continuous-state/action MDP control setting and evaluates the Natural Actor-Critic on cart-pole balancing and robotic motor-primitive tasks, including point-to-point movement and a baseball swing. Inputs and outputs are described as state/action or movement-plan variables evolving over time, which supports 1D (t) temporal structures. Dynamics range from open-ended interaction in the general MDP/cart-pole setting to capped episodic trajectories for motor primitives. Attention is static and state is constructed via value-function/critic structures, as inferred from the actor-critic formulation.

## Evidence
### Task: Control (continuous-state/action MDP policy optimization)
- "underlying control problem is a Markov decision process (MDP)" (Section 2.1)
- "continuous state set  $\mathbb{X} = \mathbb{R}^n$ , and a continuous action set  $\mathbb{U} = \mathbb{R}^m$" (Section 2.1)
- "At any state  $x_t \in \mathbb{X}$  at time t, the actor will choose an action  $u_t \in \mathbb{U}$" (Section 2.1)
- "\sum_{t=0}^{\infty} \gamma^{t} r_{t}" (Section 2.1)
- "state-value function  $V^{\pi}(x) = \phi(x)^{\mathrm{T}} v$" (Section 2.1)
- Inference: 1D (t) inputs/outputs and Open dynamics inferred from discrete-time indexing $x_t, u_t$ and infinite-horizon sums. Static attention inferred because actions are chosen from $\pi(u_t|x_t)$ without runtime selection. Constructed state inferred from value-function approximation $V^{\pi}(x) = \phi(x)^T v$. (Section 2.1)

### Task: Cart-Pole Balancing
- "Cart-Pole Balancing is a well-known benchmark for reinforcement learning." (Section 4.1)
- "The resulting state is given by  $\mathbf{x} = [\mathbf{x}, \dot{\mathbf{x}}, \theta, \dot{\theta}]^{\mathrm{T}}$ , and the action  $\mathbf{u} = F$ ." (Section 4.1)
- "The system is treated as if it was sampled at a rate of h = 60 Hz" (Section 4.1)
- Inference: 1D (t) inputs/outputs and Open dynamics inferred from time-sampled state/action sequences with no fixed horizon. Static attention inferred from the policy form $\pi(u|x)$, and constructed state inferred from the critic/value-function formulation. (Sections 2.1, 4.1)

### Task: Point-to-point motor primitive learning
- "optimizing nonlinear dynamic motor primitives for robotics." (Section 4.2)
- "representing movement plans  $(\mathbf{q}_d, \dot{\mathbf{q}}_d)$  for the degrees of freedom (DOF) robot systems" (Section 4.2)
- "The system in Eq. (13) is a point-to-point movement" (Section 4.2)
- "r_k(x_{0:N}, u_{0:N}) = \sum_{i=0}^{N}" (Section 4.2)
- Inference: 1D (t) inputs/outputs inferred because movement plans and rewards are defined over time indices; Capped dynamics inferred from finite-horizon $0{:}N$ episodic sums and movement duration. Output labeling inferred from Eq. (13) defining $\dot{q}_{d,k}$ as the movement-plan derivative. Static attention and constructed state inferred from the actor-critic policy/value-function structure. (Sections 2.1, 4.2)

### Task: Baseball swing (hitting a T-ball)
- "hitting a T-ball with a baseball bat" (Section 4)
- "planning of these motor primitives for a seven DOF robot task." (Section 4.2)
- "The task of the robot is to hit the ball properly so that it flies as far as possible." (Section 4.2)
- "performance of a baseball swing task when using the motor primitives for learning." (Fig. 5 caption)
- Inference: Input/output trajectories and capped episodic dynamics inferred from the statement that the baseball task uses the same motor-primitive setup as the point-to-point movement task. Static attention and constructed state inferred from the actor-critic policy/value-function structure used throughout. (Sections 2.1, 4.2)
