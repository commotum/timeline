# TD-Gammon, A Self-Teaching Backgammon Program, Achieves Master-Level Play (Not specified in the paper)
Source: TD-Gammon, a Self-Teaching Backgammon Program, Achieves Master-Level Play.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prediction (backgammon outcome/equity estimation) | Backgammon board positions in self-play trajectories | 2D (x, y); 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Expected game outcome / equity estimate | 0D (inferred) | Fixed (inferred) |
| Control (backgammon move selection during play) | Current backgammon board position and legal move set over game time steps | 2D (x, y); 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Selected move/action at each time step | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper covers two tightly coupled intents: predicting expected game outcome from board states and using those estimates to control move selection in backgammon play. The OCR text supports board-state inputs and temporal game sequences, giving a mixed dimensional view of 2D board structure with 1D temporal progression. Dynamics are open for gameplay trajectories because game length is variable and not capped in the described interface, while value prediction output is a fixed scalar per evaluated position. Attention and state dynamics are inferred from fixed board encodings plus runtime move/search selection driven by learned equity estimates.

## Evidence
### Task: Prediction (backgammon outcome/equity estimation)
- "the network observes a sequence of board positions

 $x_1, x_2, ..., x_f$  leading to a final reward signal z determined by the outcome of the game." (Main text)
- "the move selected at each time step was the move that maximized the network's estimate of expected outcome." (Main text)
- "given only a \"raw\" description of the board state" (Abstract)
- Inference: `2D (x, y); 1D (t)` is inferred from board-state representation plus explicit game-time sequences (`x_1 ... x_f`). `Open` input dynamics is inferred because "games often last several hundred or even several thousand time steps," indicating variable, uncapped trajectory length (Main text). `0D` and `Fixed` output are inferred because the model uses an "estimate of expected outcome" as a single equity/value per evaluated position (Main text). `Static` attention and `Constructed` state are inferred from fixed board encodings plus a learned evaluation function used for decisions.

### Task: Control (backgammon move selection during play)
- "learning strategies for the game of backgammon." (Main text)
- "the move selected at each time step was the move that maximized the network's estimate of expected outcome." (Main text)
- "with random play on both sides, games often last several hundred or even several thousand time steps." (Main text)
- "Version 1.0 used 1-ply search for move selection; versions 2.0 and 2.1 used 2-ply search." (Table 1 caption)
- Inference: `2D (x, y); 1D (t)` is inferred from board-based play over time steps. `Open` in/out dynamics are inferred from variable game duration and sequential move production. `Dynamic` attention is inferred because move selection and ply search are runtime-dependent over legal continuations. `Constructed` state is inferred because learned equity estimates and search structure are used as first-class decision state beyond raw board input.
