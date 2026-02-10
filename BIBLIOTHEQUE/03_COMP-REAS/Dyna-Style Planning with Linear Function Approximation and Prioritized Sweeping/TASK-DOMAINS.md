# Dyna-Style Planning with Linear Function Approximation and Prioritized Sweeping (Not specified in the paper)
Source: Dyna-Style Planning with Linear Function Approximation and Prioritized Sweeping.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prediction (policy evaluation of state values) | Feature-vector state transitions and rewards from online interaction trajectories | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | State-value estimates via a learned linear value function | 0D (inferred) | Open (inferred) |
| Control (online action selection and planning) | Feature vectors, rewards, and action-conditioned transitions in ongoing episodes | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Discrete action choices and policy-improvement updates | 0D (inferred) | Open (inferred) |

## Summary
The paper covers two reinforcement-learning task intents: policy evaluation (value prediction) and control. Both are framed as sequential, online interaction problems over time-indexed trajectories, supporting a 1D (t) input view with open-ended dynamics. The algorithms use adaptive update selection (e.g., prioritized sweeping queues) rather than a fixed one-shot context, supporting dynamic attention. The decision process relies on learned feature representations and learned model/value parameters, supporting a constructed-state characterization.

## Evidence
### Task: Prediction (policy evaluation of state values)
- "experience consists of the time indexed stream  $s_0, a_0, r_1, s_1, a_1, r_2, s_2, \ldots$" (Section 2 Notation)
- "An important step towards finding a good policy is to estimate the value function for a given policy (policy evaluation)." (Section 2 Notation)
- "The value function is approximated as a linear function with parameter vector  $\theta \in \mathbb{R}^n$" (Section 2 Notation)
- Inference: `1D (t)`, `Open`, `Dynamic`, `Constructed`, and output-side `0D`/`Open` are inferred from the time-indexed online stream and continual planning setup ("online setting in which estimates must be available after each interaction with the world," Abstract), queue-based selective planning updates (Algorithms 2-3, Section 4), and learned internal model/value structures (`F`, `b`, `\theta`) over features rather than direct raw state access (Sections 2-4).

### Task: Control (online action selection and planning)
- "We consider the problem of efficiently learning optimal control policies and value functions over large state spaces in an online setting in which estimates must be available after each interaction with the world." (Abstract)
- "We now turn to the full case of control, in which separate models  $F_a, b_a$  are learned and are then available for each action a." (Section 5 Theory for Control)
- "a \leftarrow \arg\max_{a} \left[ b_a^{\top} \phi + \gamma \theta^{\top} F_a \phi \right]" (Algorithm 4)
- Inference: `1D (t)` and `Open` (input/output dynamics) are inferred from ongoing time-step interaction and continual control updates; `Dynamic` attention is inferred from runtime action selection plus priority-queue-driven update targeting (Algorithm 4, Section 4); `Constructed` state is inferred from learned action-conditional models (`F_a`, `b_a`) and value parameters (`\theta`) used for decision-making (Section 5).
