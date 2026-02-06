# Beyond A* : Better Planning with Transformers via Search Dynamics Bootstrapping (Not specified in the paper.)
Source: Beyond A-- Planning with Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Planning (maze navigation shortest path) | Maze grid (walls, start, goal) | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Plan (sequence of moves/positions), optionally with A* execution trace | 1D (t) | Not specified in the paper. |
| Planning (Sokoban puzzle solving) | Sokoban grid (worker, boxes, docks) | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | Plan (sequence of moves), optionally with A* execution trace | 1D (t) | Capped (inferred) |

## Summary
The paper trains Transformer models to solve symbolic planning in two 2D grid domains: maze navigation (shortest path) and Sokoban puzzle solving. Inputs are grid states, and outputs are 1D sequences representing plans, with search-augmented models also emitting A* execution traces. The Sokoban tasks use a fixed 7x7 grid and capped sequence lengths, while other dynamics are not explicitly specified; attention and state dynamics are inferred as static and direct from the encoder-decoder setup.

## Evidence
### Task: Planning (maze navigation shortest path)
- "We consider two domains: maze navigation (Figure 1(a)) and solving Sokoban puzzles (Figure 5 in Appendix C)." (Section 3 Problem Setup)
- "In maze navigation, the goal is to find the shortest path through an n-by-n maze." (Section 3 Problem Setup)
- "we express a planning task and its optimal solution plan as a sequence of words, called *tokens*." (Introduction, Our work)
- "The resulting plan is then appended to this trace." (Section 3.1 Generating execution traces of $A^*$ search)
- Inference: Attention Dynamic = Static and State Dynamic = Direct because "The encoder processes the prompt> part of a training sequence, and the decoder processes either a <trace><plan>-formatted sequence" (Appendix A).

### Task: Planning (Sokoban puzzle solving)
- "We consider two domains: maze navigation (Figure 1(a)) and solving Sokoban puzzles (Figure 5 in Appendix C)." (Section 3 Problem Setup)
- "In Sokoban, a worker can move up, down, left, or right and has to push each box onto a dock to solve the puzzle." (Section 3 Problem Setup)
- "For Sokoban, a  $7 \times 7$  grid was sampled and two additional wall cells were added as obstacles to the interior of the map." (Appendix C Dataset generation)
- "the Sokoban dataset was further sliced to only include sequences of with at most 10000 tokens." (Appendix C Dataset generation)
- "we express a planning task and its optimal solution plan as a sequence of words, called *tokens*." (Introduction, Our work)
- Inference: Attention Dynamic = Static and State Dynamic = Direct because "The encoder processes the prompt> part of a training sequence, and the decoder processes either a <trace><plan>-formatted sequence" (Appendix A). Out Dynamics = Capped because "the Sokoban dataset was further sliced to only include sequences of with at most 10000 tokens" (Appendix C Dataset generation).
