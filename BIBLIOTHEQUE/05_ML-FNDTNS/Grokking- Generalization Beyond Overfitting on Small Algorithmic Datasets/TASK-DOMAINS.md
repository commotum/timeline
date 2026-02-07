# GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS (Not specified in the paper.)
Source: Grokking- Generalization Beyond Overfitting on Small Algorithmic Datasets.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary operation table completion | tokens (a, \\circ, b, =) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | token (x \\circ y result symbol) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper studies symbolic prediction on small algorithmic datasets where each example is a fixed-form equation for a binary operation. Inputs are 1D token sequences and outputs are single discrete symbols, so the task is a fixed-size mapping rather than open-ended generation. Attention and state dynamics are inferred from the fixed equation format and the decoder-only transformer with causal attention masking.

## Evidence
### Task: binary operation table completion
- "The datasets we consider are binary operation tables of the form  $a \circ b = c$  where a, b, c are discrete symbols with no internal structure, and  $\circ$  is a binary operation." (Section 1 Introduction)
- "Training a neural network on a proper subset of all possible equations then amounts to filling in the blanks of the binary op table, much like solving a Sudoku puzzle." (Section 1 Introduction)
- "For each binary operation we constructed a dataset of equations of the form  $\langle x \rangle \langle op \rangle \langle y \rangle \langle = \rangle \langle x \circ y \rangle$ , where  $\langle a \rangle$  stands for the token corresponding to element a." (Appendix A.1.1)
- "We trained a standard decoder-only transformer Vaswani et al. (2017) with causal attention masking, and calculated loss and accuracy only on the answer part of the equation." (Appendix A.1.2)
- Inference: In Dimension, In Dynamics, Out Dimension, Out Dynamics, Attention Dynamic, and State Dynamic are inferred from the fixed-form equation tokenization and the decoder-only transformer with causal attention masking described in the Method/Appendix.
