# Relative Entropy Policy Search (2010)
Source: Relative Entropy Policy Search (REPS).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (reinforcement learning policy search) | Series of states, actions, rewards, and state features `\phi(s)` from sampled transitions `(s_i, a_i, s'_i, r_i)` | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Policy `\pi(a|s)` (action distribution conditioned on state) | 1D (t) (inferred) | Fixed (inferred) |
| Primitive selection in robot table tennis | External stimulus with a set of motor primitives and reinforcement signals | Not specified in the paper. | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Task-appropriate motor primitive selection via a gating network | 0D (inferred) | Not specified in the paper. |

## Summary
The paper primarily covers reinforcement learning control via policy search, where sampled transition experience and state features are used to produce improved stochastic policies. The OCR text supports temporal interaction and iterative sampling, with inferred open input dynamics and a constructed internal state through learned value-function parameters. A second, distinct application task is primitive selection in robot table tennis using a gating network trained with reinforcement learning. For that application, most explicit dimension/dynamics details are not specified in the paper.

## Evidence
### Task: Control (reinforcement learning policy search)
- "Relative entropy policy search (REPS) aims at finding the optimal policy that maximizes the expected return based on all observed series of states, actions and rewards." (Section Motivation)
- "**Sampling:** Obtain samples  $(s_i, a_i, s'_i, r_i)$ , e.g., by observing another policy or being on-policy." (Section Relative Entropy Policy Search, Table 1)
- "**Output:** Policy  $\pi(a|s)$ ." (Section Relative Entropy Policy Search, Table 1)
- "Note that N is not a fixed number but may change after every iteration." (Section Policy Iteration with REPS, Table 2 caption)
- "Here, the value function  $V_s(\theta) = \theta^T \phi_s$  is determined by minimizing" (Section Relative Entropy Policy Search Method)
- Inference: `1D (t)` input/output is inferred from the explicit phrase "series of states, actions and rewards" and repeated transition sampling. `Open` input dynamics is inferred because the method repeatedly samples and "N is not a fixed number." `Static` attention is inferred because the method optimizes over the provided sampled distribution/features rather than runtime retrieval selection. `Constructed` state is inferred from the learned value function representation `V_s(\theta) = \theta^T \phi_s`.

### Task: Primitive selection in robot table tennis
- "A key problem in a skill learning system with multiple motor primitives (e.g., many different forehands, backhands, smashes, etc.) is the selection of task-appropriate primitives triggered by an external stimulus." (Section Primitive Selection in Robot Table Tennis)
- "Here, we have generated a large set of motor primitives that are triggered by a gating network that selects and generalizes among them similar to a mixture of experts." (Section Primitive Selection in Robot Table Tennis)
- "REPS improves the gating network by reinforcement learning where any successful hit results as a reward of +1 and for failures no reward is given." (Section Primitive Selection in Robot Table Tennis)
- Inference: `Dynamic` attention is inferred from the gating network that "selects and generalizes" among primitives at runtime. `Constructed` state is inferred because behavior depends on a learned gating network over a primitive set. `0D` output is inferred as each gating decision selects a discrete primitive; broader dimensional/dynamic interface constraints are not explicitly specified.
