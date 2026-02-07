# ORDER MATTERS: SEQUENCE TO SEQUENCE FOR SETS (Not specified in the paper.)
Source: Order Matters- Sequence to sequence for sets.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sorting numbers | Unordered floating-point numbers (set) | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Sorted order of the numbers (sequence of indices) | 1D (t) | Not specified in the paper. |
| Language modeling | Word sequences (tokens) | 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Word sequences (tokens) | 1D (t) | Not specified in the paper. |
| Constituency parsing | Sentence (tokens) | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Parse tree linearized as a sequence | 1D (t) | Not specified in the paper. |
| Joint probability modeling (graphical models) | Random variables (set/sequence) | 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Random variables (sequence) | 1D (t) | Not specified in the paper. |

## Summary
Across experiments, the paper covers four tasks: language modeling, constituency parsing, sorting numbers, and joint probability modeling for graphical models. Inputs and outputs are 1D sequences or sets of elements; sorting explicitly uses unordered input sets and parsing outputs linearized parse trees. Attention mechanisms are described for the sorting and parsing setups, while no attention mechanism is described in the language modeling or graphical-model experiments. The paper does not specify fixed bounds on input or output sizes.

## Evidence
### Task: Sorting numbers
- "for the task of sorting numbers: given N unordered random floating point numbers between 0 and 1, we return them in a sorted order." (Section 4.4 Sorting Experiment)
- "the *Process* module is an attention mechanism over the read numbers" (Section 4.4 Sorting Experiment)
- "implemented as T steps over an LSTM with no input nor output" (Section 4.4 Sorting Experiment)
- "produce indices in the input numbers with a pointer network" (Section 4.4 Sorting Experiment)
- Inference: Attention Dynamic = Dynamic because the process module uses an attention mechanism; State Dynamic = Constructed because the model uses an LSTM process block. (Section 4.4 Sorting Experiment)

### Task: Language modeling
- "We trained medium sized LSTMs with large amounts of regularization" (Section 5.1.1 Language Modeling)
- "to estimate probabilities over sequences of words." (Section 5.1.1 Language Modeling)
- Inference: Attention Dynamic = Static because no attention mechanism is described for this experiment; State Dynamic = Constructed because the model is an LSTM. (Section 5.1.1 Language Modeling)

### Task: Constituency parsing
- "The task of constituency parsing consists in producing a parse tree given a sentence." (Section 5.1.2 Parsing)
- "a sentence encoder LSTM followed by a decoder LSTM trained to generate a depth first traversal encoding of the parse tree, using an attention mechanism." (Section 5.1.2 Parsing)
- Inference: Attention Dynamic = Dynamic because an attention mechanism is used; State Dynamic = Constructed because the encoder/decoder are LSTMs. (Section 5.1.2 Parsing)

### Task: Joint probability modeling (graphical models)
- "Let us consider the joint probability of a set of T random variables  $P(y_1, y_2, \ldots, y_T)$ ." (Section 5.1.4 Graphical Models)
- "We generated *star-like* graphical models over random variables" (Section 5.1.4 Graphical Models)
- "we trained two LSTMs for 10,000 mini-batch iterations to model the joint probability" (Section 5.1.4 Graphical Models)
- Inference: Attention Dynamic = Static because no attention mechanism is described; State Dynamic = Constructed because the model is an LSTM. (Section 5.1.4 Graphical Models)
