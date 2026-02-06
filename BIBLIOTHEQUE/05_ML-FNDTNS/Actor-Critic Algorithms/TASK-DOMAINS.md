# Actor-Critic Algorithms (Not specified in the paper.)
Source: Actor-Critic Algorithms.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (randomized stationary policy) | state (x) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | action distribution over A | 0D (inferred) | Fixed (inferred) |
| prediction (q-function value estimation) | state-action pair (x, u) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | q-function value | 0D (inferred) | Fixed (inferred) |

## Summary
The paper addresses reinforcement learning control in finite-state/action Markov decision processes via randomized stationary policies and a critic that estimates q-function values. The inputs and outputs are point-like state or state-action objects with fixed dynamics under the stated finite S and A assumptions, while attention is static to the current state/action and internal state is constructed through learned parameter vectors and traces. These characterizations are inferred from the algorithm descriptions rather than explicitly labeled in the paper.

## Evidence
### Task: control (randomized stationary policy)
- "We propose and analyze a class of actor-critic algorithms for simulation-based optimization of a Markov decision process over a parameterized family of randomized stationary policies." (Abstract)
- "A randomized stationary policy (RSP) is a mapping  $\mu$  that assigns to each state x a probability distribution over the action space A." (Section 2)
- Inference: Treated dimensions/dynamics as 0D/Fixed because the paper assumes a "finite state space S, and finite action space A," and the policy is "parameterized in terms of a vector  $\theta$ "; attention is static and state is constructed due to the parameterized policy. (Section 2)

### Task: prediction (q-function value estimation)
- "we define the q-function  $q_{\theta}: S \times A \to \mathbb{R}$ , by" (Section 2)
- "the job of the critic is to compute an approximation of the projection  $\Pi_{\theta}q_{\theta}$  of  $q_{\theta}$  onto  $\Psi_{\theta}$ ." (Section 3)
- Inference: Marked dimensions/dynamics as 0D/Fixed because the paper assumes a "finite state space S, and finite action space A," and inferred constructed state from "Along with the parameter vector r, the critic stores some auxiliary parameters." (Section 2; Section 3)
