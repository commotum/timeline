# Prioritized Experience Replay (Not specified in the paper.)
Source: Prioritized Experience Replay.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (inferred) | states; visual observations (Atari) (inferred) | 0D (inferred); 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |
| classification | MNIST digit samples (inferred) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | digit class labels (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper applies prioritized replay to reinforcement learning control in Atari games and the Blind Cliffwalk toy environment, and extends the idea to supervised MNIST digit classification. The RL setting consumes states (including vision-based observations for Atari) and outputs actions, while the supervised setting consumes digit samples and outputs class labels. Input dimensionalities are reported as 0D and 2D (x, y), with fixed dynamics; attention and state dynamics are marked static/direct where inferred from the algorithmic description.

## Evidence
### Task: control (inferred)
- "We use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm that achieved human-level performance across many Atari games." (Abstract)
- "the collection of Atari benchmarks (Bellemare et al., 2012) with their end-to-end RL from vision setup" (Section 4 Atari Experiments)
- "Observe S_0 and choose A_0 \sim \pi_{\theta}(S_0)" (Algorithm 1)
- "there are two actions, a 'right' and a 'wrong' one" (Figure 1 caption, Section 3.1)
- "progresses through a sequence of *n* states" (Figure 1 caption, Section 3.1)
- Inference: Labeled the task as control and set static/direct attention/state plus 0D/2D inputs and fixed dynamics based on the RL state-action formulation and vision-based Atari setup.

### Task: classification
- "class-imbalanced variant of the classical MNIST digit classification problem" (Section 6 Extensions)
- "we removed 99% of the samples for digits 0, 1, 2, 3, 4 in the training set" (Section 6 Extensions)
- Inference: Treated MNIST digit classification as 2D image inputs with fixed size and 0D class-label outputs, with static/direct processing implied by the supervised setup.
