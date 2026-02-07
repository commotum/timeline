# Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (Not specified in the paper.)
Source: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (Atari game playing) | last 32 RGB frames (96x96) and last 32 actions (Atari observations) | 3D (x, y, t); 1D (t) | Fixed | Dynamic (inferred) | Constructed | action-selection policy / action | 0D | Fixed |
| control (Go game playing) | last 8 board states (Go) | 3D (x, y, t) | Fixed | Dynamic (inferred) | Constructed | move policy / action | 2D (x, y) | Fixed |
| control (chess game playing) | last 100 board states (chess) | 3D (x, y, t) | Fixed | Dynamic (inferred) | Constructed | move policy / action | 2D (x, y) | Fixed |
| control (shogi game playing) | last 8 board states (shogi) | 3D (x, y, t) | Fixed | Dynamic (inferred) | Constructed | move policy / action | 2D (x, y) | Fixed |

## Summary
MuZero is evaluated on control/game-playing tasks in Atari and the board games Go, chess, and shogi. Inputs are fixed-length histories of visual observations: board states for the board games and 32-frame RGB histories plus action histories for Atari, yielding spatiotemporal inputs (and a 1D action history for Atari). Outputs are discrete move/action policies tied to fixed action spaces, while planning with MCTS implies dynamic attention and the model's hidden-state updates indicate constructed state.

## Evidence
### Task: control (Atari game playing)
- "When evaluated on 57 different Atari games - the canonical video game environment for testing AI techniques," (Abstract)
- "For Atari, the input of the representation function includes the last 32 RGB frames at resolution 96x96 along with the last 32 actions" (Appendix E Network Input)
- "and transforms it into a hidden state." (Introduction)
- "In Atari, an action is encoded as a one hot vector which is tiled appropriately into planes." (Appendix E Network Input)
- Inference: Marked Attention Dynamic as Dynamic because "A Monte-Carlo Tree Search is performed at each timestep t" indicates runtime selection during planning. (Figure 1 caption)

### Task: control (Go game playing)
- "We applied the MuZero algorithm to the classic board games Go, chess and shogi" (Section 4 Results)
- "In Go and shogi we encode the last 8 board states as in *AlphaZero*;" (Appendix E Network Input)
- "and transforms it into a hidden state." (Introduction)
- "In Go, a normal action (playing a stone on the board) is encoded as an all zero plane," (Appendix E Network Input)
- Inference: Marked Attention Dynamic as Dynamic because "A Monte-Carlo Tree Search is performed at each timestep t" indicates runtime selection during planning. (Figure 1 caption)

### Task: control (chess game playing)
- "We applied the MuZero algorithm to the classic board games Go, chess and shogi" (Section 4 Results)
- "in chess we increased the history to the last 100 board states to allow correct prediction of draws." (Appendix E Network Input)
- "and transforms it into a hidden state." (Introduction)
- "The first one-hot plane encodes which position the piece was moved from." (Appendix E Network Input)
- Inference: Marked Attention Dynamic as Dynamic because "A Monte-Carlo Tree Search is performed at each timestep t" indicates runtime selection during planning. (Figure 1 caption)

### Task: control (shogi game playing)
- "We applied the MuZero algorithm to the classic board games Go, chess and shogi" (Section 4 Results)
- "In Go and shogi we encode the last 8 board states as in *AlphaZero*;" (Appendix E Network Input)
- "and transforms it into a hidden state." (Introduction)
- "We use the first 8 planes to indicate where the piece moved from" (Appendix E Network Input)
- Inference: Marked Attention Dynamic as Dynamic because "A Monte-Carlo Tree Search is performed at each timestep t" indicates runtime selection during planning. (Figure 1 caption)
