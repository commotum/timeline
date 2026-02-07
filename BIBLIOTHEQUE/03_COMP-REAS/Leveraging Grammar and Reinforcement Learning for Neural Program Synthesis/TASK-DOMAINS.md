# LEVERAGING GRAMMAR AND REINFORCEMENT LEARNING FOR NEURAL PROGRAM SYNTHESIS (Not specified in the paper.)
Source: Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis (generate Karel DSL program from IO examples) | Input/output grid pairs (IO examples) | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | Program tokens (Karel DSL program) | 1D (t) | Not specified in the paper. |

## Summary
The paper frames program synthesis as generating Karel DSL programs from small sets of input/output grid examples. Inputs are fixed-size 2D grid states (18x18 with 16 channels) provided as multiple IO pairs, while outputs are token sequences representing programs. The model processes all IO pairs with max-pooled decoder states and builds learned embeddings, which supports Static attention and Constructed state (both inferred); output length dynamics are not specified.

## Evidence
### Task: Program synthesis (generate Karel DSL program from IO examples)
- "Program synthesis is the task of automatically generating a program consistent with a specification." (Abstract)
- "Our goal is to learn a synthesizer  $\sigma$  that, given a set of input/output examples produces a program:" (Section 3.1 Program Synthesis Formulation)
- "Our goal is to learn to generate a program in the Karel DSL given a small set of input and output grids." (Section 6.1 The Domain: Karel)
- "In our Karel environment, states are grids describing the presence of objects." (Section 3.1 Program Synthesis Formulation)
- "We represent the input and output elements as grids where each cell in the grid is a vector with 16 channels" (Section 6.1 The Domain: Karel)
- "The state of the grid word are represented as a  $16\times18\times18$  tensor." (Section C Experiments Hyperparameters)
- "6 are kept for each program" (Section 6.1 The Domain: Karel)
- "The first 5 samples serve as the specification, and the sixth one is kept as held-out test pair." (Section 6.1 The Domain: Karel)
- "Each program is represented by a sequence of tokens  $\lambda = [s_1, s_2, ..., s_L]$  where each token comes from an alphabet  $\Sigma$ ." (Section 3.2 Neural Program Synthesis Architecture)
- "One decoder LSTM is run for each of the IO pairs, all using the same weights." (Section 3.2 Neural Program Synthesis Architecture)
- "The probability of the next token is defined as the Softmax of a linear layer over the max-pooled hidden state" (Section 3.2 Neural Program Synthesis Architecture)
- "Each pair is encoded independently by a convolutional neural network (CNN) to generate a joint embedding." (Section 3.2 Neural Program Synthesis Architecture)
- Inference: Attention Dynamic is Static because the model runs a decoder over every IO pair and max-pools across all pairs, rather than selecting inputs at runtime; State Dynamic is Constructed because IO pairs are encoded into learned embeddings and LSTM states drive generation. (Section 3.2 Neural Program Synthesis Architecture)
