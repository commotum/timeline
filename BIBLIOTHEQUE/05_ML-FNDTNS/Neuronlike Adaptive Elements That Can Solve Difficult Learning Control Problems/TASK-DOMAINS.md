# Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems (1983 (inferred))
Source: Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (pole balancing) | cart-pole system state vector | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | control force/action to cart (left/right) | 1D (t) (inferred) | Open (inferred) |
| prediction (reinforcement expectation) | current cart-pole state vector | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | improved/internal reinforcement signal (reinforcement prediction) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper applies the ASE/ACE system to a sequential pole-balancing control task driven by cart-pole state vectors and discrete left/right control forces, and it includes an adaptive critic that predicts reinforcement signals from state. The inputs and outputs are time-indexed streams, with no explicit fixed length, so the task interfaces are best described as 1D (t) with Open dynamics (inferred). Attention is Static and state is Constructed (inferred) based on the fixed state-vector inputs and the explicit use of memory traces and reinforcement predictions.

## Evidence
### Task: control (pole balancing)
- "The task is to balance a pole that is hinged to a movable cart by applying forces to the cart's base." (Abstract)
- "At each time step, the controller receives a vector giving the cart-pole system's state at that instant." (Section V)
- "The controller can apply an impulsive \"left\" or \"right\" force F of fixed magnitude to the cart at discrete time intervals." (Section V)
- "ASE's output determines force applied to cart." (Fig. 2)
- "At each synapse of the ASE are both a long-term memory trace" (Section VII)
- "and a shortterm memory trace that is required to update the long-term trace." (Section VII)
- Inference: In/Out Dimension and Dynamics marked 1D (t) / Open because the controller operates "at each time step" with actions at "discrete time intervals" (Section V). Attention marked Static because the full state vector is provided without a selection mechanism (Section V). State marked Constructed because the ASE maintains long-term and short-term memory traces (Section VII).

### Task: prediction (reinforcement expectation)
- "The ACE receives the externally supplied reinforcement signal" (Section VIII)
- "on the basis of the current cart-pole state vector, an improved reinforcement signal that it sends to the ASE." (Section VIII)
- "the job of the ACE is to store in each box a prediction or expectation of the reinforcement" (Section VIII)
- "Each nonreinforcement input pathway i has a weight with real value v_i(t) at time t." (Section VIII)
- Inference: In/Out Dimension and Dynamics marked 1D (t) / Open because the ACE computes signals indexed by time t from the ongoing cart-pole state stream (Section VIII). Attention marked Static because the ACE uses the provided state vector without runtime selection (Section VIII). State marked Constructed because the ACE stores predictions and maintains learned weights (Section VIII).
