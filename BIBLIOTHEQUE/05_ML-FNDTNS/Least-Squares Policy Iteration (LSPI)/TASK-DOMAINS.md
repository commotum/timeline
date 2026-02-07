# Least-Squares Policy Iteration (Not specified in the paper)
Source: Least-Squares Policy Iteration (LSPI).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (chain walk navigation) | state (chain position/state number) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | action (left/right) | 0D (inferred) | Fixed (inferred) |
| control (inverted pendulum balancing) | state (theta, theta_dot) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | action (left/right/no force on cart) | 0D (inferred) | Fixed (inferred) |
| control (bicycle balancing and riding to target) | state (theta, theta_dot, omega, omega_dot, omega_ddot, psi) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | action (handlebar torque tau, rider displacement v; 5 discrete actions) | 0D (inferred) | Fixed (inferred) |

## Summary
LSPI is evaluated on three control domains: chain-walk navigation, inverted pendulum balancing, and bicycle balancing/riding to a target. Inputs are per-step state descriptions (discrete state IDs or continuous state vectors) and outputs are discrete control actions, giving point-like interfaces with fixed size (0D/Fixed; inferred). The paper describes action selection via a learned linear state-action value function, so attention is static and decision state is constructed (inferred).

## Evidence
### Task: control (chain walk navigation)
- "a chain with 4 states (numbered from 1 to 4)" (Section 9.1 Chain Walk)
- "There are two actions available, \"left\" (L) and \"right\" (R)." (Section 9.1 Chain Walk)
- "the state-action value function is approximated by a linear architecture" (Section 4 Reinforcement Learning and Approximate Policy Iteration)
- Inference: Marked In/Out Dimension as 0D and Dynamics as Fixed because the task defines discrete states and a fixed action set; marked Attention Static and State Constructed because action selection uses the learned value-function representation (Sections 9.1 and 4).

### Task: control (inverted pendulum balancing)
- "requires balancing a pendulum of unknown length and mass at the upright position" (Section 9.2 Inverted Pendulum)
- "Three actions are allowed: left force LF (-50 Newtons), right force RF (+50 Newtons), or no force NF (0 Newtons)." (Section 9.2 Inverted Pendulum)
- "The state space of the problem is continuous and consists of the vertical angle  $\theta$  and the angular velocity  $\dot{\theta}$  of the pendulum." (Section 9.2 Inverted Pendulum)
- "the state-action value function is approximated by a linear architecture" (Section 4 Reinforcement Learning and Approximate Policy Iteration)
- Inference: Marked In/Out Dimension as 0D and Dynamics as Fixed because the task specifies a fixed-size state vector and a fixed discrete action set; marked Attention Static and State Constructed because action selection uses the learned value-function representation (Sections 9.2 and 4).

### Task: control (bicycle balancing and riding to target)
- "to learn to balance and ride a bicycle to a target position located 1 km away" (Section 9.3 Bicycle Balancing and Riding)
- "The state description is a six-dimensional real-valued vector  $(\theta, \dot{\theta}, \omega, \dot{\omega}, \ddot{\omega}, \psi)$" (Section 9.3 Bicycle Balancing and Riding)
- "The actions are the torque  $\tau$  applied to the handlebar" (Section 9.3 Bicycle Balancing and Riding)
- "and the displacement of the rider v (discretized to  $\{-0.02, 0, +0.02\}$ )." (Section 9.3 Bicycle Balancing and Riding)
- "the state-action value function is approximated by a linear architecture" (Section 4 Reinforcement Learning and Approximate Policy Iteration)
- Inference: Marked In/Out Dimension as 0D and Dynamics as Fixed because the task specifies a fixed-size state vector and discrete action choices; marked Attention Static and State Constructed because action selection uses the learned value-function representation (Sections 9.3 and 4).

## CSV Output (required)
CSV written to `/home/jake/Developer/timeline/BIBLIOTHEQUE/05_ML-FNDTNS/Least-Squares Policy Iteration (LSPI)/.TASK-DOMAINS.csv.tmp.d85ed1a3f53e47feba952383398bda56`.
