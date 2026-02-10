# RobustFill: Neural Program Learning under Noisy I/O (2017)
Source: RobustFill- Neural Program Learning under Noisy I-O.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis for string transformation | Observed I/O string examples `(I_1,O_1),...,(I_n,O_n)` | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | DSL program `P` (token sequence) | 1D (t) (inferred) | Open (inferred) |
| Program induction for string transformation | Observed I/O string examples `(I_1,O_1),...,(I_n,O_n)` plus assessment input string `I^y` | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Assessment output string `O^y` (character sequence) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper covers Programming by Example string transformation with two distinct model intents: generating a DSL program (synthesis) and generating target strings directly (induction). In both cases, the data objects are character/token sequences, so the justified dimensionality is 1D (t). The model interface is described as variable-length and sequence-generative, supporting Open dynamics for inputs and outputs. Both approaches use attentional RNNs and learned hidden representations, supporting Dynamic attention and Constructed state.

## Evidence
### Task: Program synthesis for string transformation
- "In the program synthesis approach, we train a neural model which takes  $(I_1, O_1), ..., (I_n, O_n)$  as input and generates P as output, token-by-token." (Section 3.1. Problem Formulation)
- "In all cases, the InStr and OutStr are processed at the character level, so the input to I and O are character embeddings." (Section 4.1. Single-Example Representation)
- "The inputs and targets for the P layer is the source-codeorder linearization of the program." (Section 4.1. Single-Example Representation)
- Inference: `1D (t)`, `Open`, `Dynamic`, `Constructed`, and output `1D (t)`/`Open` are inferred from sequence processing and runtime attention over variable-size example sets: "the input is a variable-length, unordered set of sequence pairs" (Section 4. Program Synthesis Model Architecture) and the attention formulation in "Double attention" (Section 4.2).

### Task: Program induction for string transformation
- "we can train a neural network which takes as input a set of n observed examples  $(I_1, O_1), ...(I_n, O_n)$  as well an unpaired InStr,  $I^y$ , and generates the corresponding OutStr,  $O^y$ ." (Section 6. Program Induction Results)
- "The induction model generates  $O^y$  directly by sequentially predicting each character." (Section 6. Program Induction Results)
- "There is an additional LSTM to encode  $I^y$ . The decoder layer  $O^y$  uses double attention on  $O_i$  and  $I^y$ ." (Section 6.1. Comparison of Induction and Synthesis Models)
- Inference: `1D (t)`, `Open`, `Dynamic`, `Constructed`, and output `1D (t)`/`Open` are inferred because inputs/outputs are character sequences and decoding is sequential with learned recurrent/attentional state over variable-sized observed examples.
