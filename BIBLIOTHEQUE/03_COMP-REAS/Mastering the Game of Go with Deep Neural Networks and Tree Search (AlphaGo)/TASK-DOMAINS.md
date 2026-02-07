# Mastering the game of Go with deep neural networks and tree search (2016)
Source: Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (move prediction) | board position / board state (Go) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | probability distribution over legal moves (probability map over board) | 2D (x, y) (inferred) | Fixed (inferred) |
| prediction (game outcome / value) | board position / board state (Go) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | scalar value / expected outcome (winner) | 0D (inferred) | Fixed (inferred) |
| control (search-based move selection) | root board position / state (Go) | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | selected move/action | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers Go move prediction via policy networks, game-outcome/value prediction via a value network, and search-based move selection via MCTS. Inputs are Go board positions represented as fixed 19x19 grids (2D), while outputs are either a 2D move-probability map or a 0D scalar outcome, and the search outputs a discrete move. Attention and state are static/direct for the feedforward networks but dynamic/constructed for MCTS due to its evolving search tree (inferred).

## Evidence
### Task: classification (move prediction)
- "We trained the policy network  $p_{\sigma}$  to classify positions according to expert moves played in the KGS data set." (Policy network: classification)
- "A final softmax layer outputs a probability distribution over all legal moves a." (Supervised learning of policy networks)
- "We pass in the board position as a  $19 \times 19$  image" (ARTICLE)
- "represented by a probability map over the board." (Figure 2b, Schematic representation)
- Inference: In/Out Dimension and Fixed/Static/Direct dynamics are inferred from the fixed 19x19 board image input and probability map output; no runtime attention or persistent state is described. (ARTICLE; Supervised learning of policy networks; Figure 2b)

### Task: prediction (game outcome / value)
- "Finally, we train a value network  $\nu_{\theta}$  that predicts the winner of games played by the RL policy network against itself." (ARTICLE)
- "outputs a scalar value  $v_{\theta}(s')$  that predicts the expected outcome in position s'." (Figure 2b, Schematic representation)
- "The input to the value network is also a  $19 \times 19 \times 48$  image stack" (Neural network architecture)
- Inference: In/Out Dimension and Fixed/Static/Direct dynamics are inferred from the fixed 19x19 image input and single scalar output with no described runtime selection or persistent state. (Neural network architecture; Figure 2b)

### Task: control (search-based move selection)
- "AlphaGo combines the policy and value networks in an MCTS algorithm (Fig. 3) that selects actions by lookahead search." (Searching with policy and value networks)
- "Once the search is complete, the algorithm chooses the most visited move from the root position." (Searching with policy and value networks)
- "(s, a) of the search tree stores an action value Q(s, a), visit count N(s, a), and prior probability P(s, a)." (Searching with policy and value networks)
- Inference: Input/Output dimensions and Fixed dynamics are inferred from Go board state/action framing, while Dynamic attention and Constructed state are inferred from search-tree traversal and stored statistics. (Searching with policy and value networks; ARTICLE)

## CSV Output (required)
`BIBLIOTHEQUE/03_COMP-REAS/Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo)/.TASK-DOMAINS.csv.tmp.a65124153f1a47a6b4b800e847b85c6a`
