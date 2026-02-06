# A Machine Learning Approach That Beats Large Rubik's Cubes The CayleyPy Project (Not specified in the paper.)
Source: A Machine Learning Approach That Beats Large Rubik's Cubes.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Planning / pathfinding | Rubik's cube state (graph node permutation vector) | 1D (t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Path / sequence of moves to solved state | 1D (t) (inferred) | Capped (inferred) |
| Prediction (diffusion distance regression) | Graph node feature vector v (permutation vector) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Scalar diffusion distance / steps k | 0D (inferred) | Fixed (inferred) |

## Summary
The paper describes a planning/pathfinding task on Rubik's cube (Cayley) graphs that outputs solution paths, supported by a neural network that predicts diffusion distance from a state to the solved node. Inputs are fixed-length permutation vectors (1D), while outputs are either scalar distances (0D) or variable-length action sequences (1D). The beam-search solver uses dynamic selection over candidate nodes and constructs search state, while the distance predictor is a direct, static mapping from state vectors to scalar targets.

## Evidence
### Task: Planning / pathfinding
- "one needs to plan a sequence of actions to transit between two given states." (Section I. INTRODUCTION)
- "The task is to find a path from any given node to this selected node." (Section II.A)
- "it is just the vector describing the permutation p of l-symbols" (Section II.A)
- "We then select the W nodes closest to the destination according to the neural network." (Section II.A)
- "scramble was considered unsolved if the path to the solved state was not found in 200 beam search steps." (Section IV.D)
- Inference: In/Out Dimension and In Dynamics inferred from the fixed-length permutation vector representation and the "sequence of actions" phrasing; Attention and State Dynamics inferred from beam search selecting and iterating over top-W nodes; Out Dynamics inferred from the 200-step cap described above.

### Task: Prediction (diffusion distance regression)
- "pairs (v, k), where v represents the vector corresponding to the node and k is the number of steps required" (Section II.A)
- "v serves as the 'feature vector' (the input for the neural network), and k represents the 'target' (the output the network needs to predict)." (Section II.A)
- "neural network's predictions for a given node v estimate the diffusion distance from v to the selected destination node." (Section II.A)
- Inference: In Dimension/In Dynamics inferred from the vector input v; Out Dimension/Out Dynamics inferred from scalar step count k; Attention Dynamic and State Dynamic inferred as static/direct because the model maps v directly to k without any runtime selection or persistent state described.
