# Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (Not specified in the paper.)
Source: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Move probability and outcome value estimation for chess | Chess board position state (T=8-step history planes; N x N image stack) | 3D (x, y, t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Move probability vector (policy) over legal actions; scalar expected outcome value | 1D (t) (inferred); 0D (inferred) | Fixed (inferred) |
| Move probability and outcome value estimation for shogi | Shogi board position state (T=8-step history planes; N x N image stack) | 3D (x, y, t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Move probability vector (policy) over legal actions; scalar expected outcome value | 1D (t) (inferred); 0D (inferred) | Fixed (inferred) |
| Move probability and outcome value estimation for Go | Go board position state (T=8-step history planes; N x N image stack) | 3D (x, y, t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Move probability vector (policy) over legal actions; scalar expected outcome value | 1D (t) (inferred); 0D (inferred) | Fixed (inferred) |

## Summary
AlphaZero is applied to three board-game domains (chess, shogi, Go), mapping board-state histories to move distributions and expected outcome values. Inputs are fixed-size N x N board-plane stacks with a T=8-step history, implying a grid-based spatiotemporal representation. Outputs are fixed-size move probability vectors plus a scalar value, and decision making uses MCTS with selective simulations, supporting dynamic attention and constructed state (inferred).

## Evidence
### Task: Move probability and outcome value estimation for chess
- "We applied the AlphaZero algorithm to chess, shogi, and also Go." (Section Main text)
- "The first set of features are repeated for each position in a T=8-step history." (Section Representation)
- "The input to the neural network is an  $N \times N \times (MT+L)$  image stack" (Section Representation)
- "We represent the policy  $\pi(a|s)$  by a  $8\times8\times73$  stack of planes encoding a probability distribution" (Section Representation)
- "a scalar value v estimating the expected outcome z from position s" (Section Main text)
- "Each search consists of a series of simulated games of self-play that traverse a tree from root  $s_{root}$  to leaf." (Section Main text)
- "Each simulation proceeds by selecting in each state s a move a with low visit count, high move probability and high value" (Section Main text)
- Inference: Inferred 3D input dimension and fixed input dynamics from the N x N x (MT+L) stack with a T=8-step history; inferred fixed output dynamics and 1D move distribution from the fixed policy-plane encoding; inferred dynamic attention and constructed state from the MCTS tree search and selective simulations described above.

### Task: Move probability and outcome value estimation for shogi
- "We applied the AlphaZero algorithm to chess, shogi, and also Go." (Section Main text)
- "The first set of features are repeated for each position in a T=8-step history." (Section Representation)
- "The input to the neural network is an  $N \times N \times (MT+L)$  image stack" (Section Representation)
- "The policy in shogi is represented by a  $9 \times 9 \times 139$  stack of planes" (Section Representation)
- "a scalar value v estimating the expected outcome z from position s" (Section Main text)
- "Each search consists of a series of simulated games of self-play that traverse a tree from root  $s_{root}$  to leaf." (Section Main text)
- "Each simulation proceeds by selecting in each state s a move a with low visit count, high move probability and high value" (Section Main text)
- Inference: Inferred 3D input dimension and fixed input dynamics from the N x N x (MT+L) stack with a T=8-step history; inferred fixed output dynamics and 1D move distribution from the fixed policy-plane encoding; inferred dynamic attention and constructed state from the MCTS tree search and selective simulations described above.

### Task: Move probability and outcome value estimation for Go
- "We applied the AlphaZero algorithm to chess, shogi, and also Go." (Section Main text)
- "The first set of features are repeated for each position in a T=8-step history." (Section Representation)
- "The input to the neural network is an  $N \times N \times (MT+L)$  image stack" (Section Representation)
- "using a flat distribution over  $19\times 19+1$  moves representing possible stone placements and the pass move." (Section Representation)
- "a scalar value v estimating the expected outcome z from position s" (Section Main text)
- "Each search consists of a series of simulated games of self-play that traverse a tree from root  $s_{root}$  to leaf." (Section Main text)
- "Each simulation proceeds by selecting in each state s a move a with low visit count, high move probability and high value" (Section Main text)
- Inference: Inferred 3D input dimension and fixed input dynamics from the N x N x (MT+L) stack with a T=8-step history; inferred fixed output dynamics and 1D move distribution from the flat move distribution; inferred dynamic attention and constructed state from the MCTS tree search and selective simulations described above.
