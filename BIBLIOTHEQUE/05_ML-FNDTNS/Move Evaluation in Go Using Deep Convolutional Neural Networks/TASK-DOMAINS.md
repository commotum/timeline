# Move Evaluation in Go Using Deep Convolutional Neural Networks (Not specified in the paper)
Source: Move Evaluation in Go Using Deep Convolutional Neural Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Move prediction (expert next-move classification) | Go board position feature planes (19x19) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Next-move board position / move distribution (361) | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper trains a CNN to predict expert Go moves from 19x19 board feature planes and outputs a distribution over 361 board positions. This is a fixed-size 2D grid input and output task derived from the 19x19 board representation. Attention and state are inferred as static and direct because the model is a feedforward CNN over a fixed board representation with no described dynamic selection or persistent state.

## Evidence
### Task: Move prediction (expert next-move classification)
- "The network correctly predicts the expert move in 55% of positions" (Abstract)
- "We focus on a supervised learning setup, in which the network is trained to predict expert human moves" (Section 1 Introduction)
- "Each position  $s_t$  was preprocessed into a set of  $19 \times 19$  feature planes" (Section 3 Data)
- "a move  $a_t$  is encoded as a 1 of 361 indicator for each position on the 19x19 board" (Section 3 Data)
- "two softmax distributions of size 361" (Section 4 Architecture & Training)
- Inference: In Dimension = 2D (x, y) (inferred), In Dynamics = Fixed (inferred), Out Dimension = 2D (x, y) (inferred), Out Dynamics = Fixed (inferred), Attention Dynamic = Static (inferred), and State Dynamic = Direct (inferred) because the input and output are fixed 19x19 board grids and the model is a feedforward CNN; no dynamic attention or persistent state is described. Supporting text: "Every layer operated on a  $19\times19$  input space" (Section 4 Architecture & Training) and "The output layer of the CNN was also convolutional" (Section 4 Architecture & Training)
