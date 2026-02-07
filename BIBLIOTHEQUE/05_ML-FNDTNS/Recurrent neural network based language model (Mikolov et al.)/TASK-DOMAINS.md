# Recurrent neural network based language model (Not specified in the paper.)
Source: Recurrent neural network based language model (Mikolov et al.).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next-word prediction (language modeling) | word tokens (context sequence) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | next-word probability distribution | 0D (inferred) | Fixed (inferred) |

## Summary
The paper defines a recurrent neural network language model for next-word prediction over word sequences, evaluated in speech recognition rescoring. Inputs are 1D token sequences with open-length context, and outputs are single-token probability distributions (0D) per step. The architecture implies static attention and a constructed recurrent state.

## Evidence
### Task: next-word prediction (language modeling)
- "The goal of statistical language modeling is to predict the next word in textual data given context;" (Introduction)
- "Output layer y(t) represents probability distribution of next word given previous word w(t) and context s(t-1)." (Model description)
- "Recurrent neural networks do not use limited size of context." (Model description)
- "The network has an input layer x, hidden layer s (also called context layer or state) and output layer y." (Model description)
- Inference: In Dimension 1D (t), In Dynamics Open, Attention Dynamic Static, State Dynamic Constructed, Out Dimension 0D, and Out Dynamics Fixed are inferred from the sequential next-word framing, the non-limited context claim, and the explicit recurrent state/output layer description above.
