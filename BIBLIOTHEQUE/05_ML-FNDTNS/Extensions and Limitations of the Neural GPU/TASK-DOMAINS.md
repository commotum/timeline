# EXTENSIONS AND LIMITATIONS OF THE NEURAL GPU (Not specified in the paper.)
Source: Extensions and Limitations of the Neural GPU.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Binary multi-digit addition | Binary digit sequences (two integers) | 1D (t) (inferred) | Open | Static (inferred) | Constructed (inferred) | Binary digit sequence (sum) | 1D (t) (inferred) | Open |
| Binary multi-digit multiplication | Binary digit sequences (two integers) | 1D (t) (inferred) | Open | Static (inferred) | Constructed (inferred) | Binary digit sequence (product) | 1D (t) (inferred) | Open |
| Decimal arithmetic operations (all) | Decimal digit sequences (operands) | 1D (t) (inferred) | Open | Static (inferred) | Constructed (inferred) | Decimal digit sequence (operation result) | 1D (t) (inferred) | Open |
| Binary multi-operand multiplication (3 terms) | Binary digit sequences (three integers/terms) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Binary digit sequence (product) | 1D (t) (inferred) | Open (inferred) |
| Binary arithmetic expression evaluation | Binary digit sequences with operators (+, -, *, /) and multiple operands | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Binary digit sequence (expression value) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper focuses on algorithmic arithmetic over digit sequences, covering binary addition and multiplication, decimal arithmetic operations, multi-operand multiplication, and evaluation of arithmetic expressions with operator precedence. Inputs and outputs are 1D sequences of digits (and operators for expressions), with lengths that can extend beyond training regimes, including explicit claims of arbitrary-length generalization for binary and decimal arithmetic. The Neural GPU processes variable-length sequences via convolutional recurrent computation; attention is static and state is constructed (inferred from the architecture description).

## Evidence
### Task: Binary multi-digit addition
- "The Neural GPU is a recent model that can learn algorithms such as multi-digit binary addition and binary multiplication in a way that generalizes to inputs of arbitrary length." (Abstract)
- "Integer addition is a well defined algorithm, so knowledge of its operation is sufficient to add arbitrary numbers." (Section 5 Generalization)
- Inference: In/Out Dimension are 1D (t), and Attention/State dynamics are Static/Constructed based on the variable-length sequence framing and recurrent convolutional architecture ("The Neural GPU consumes an input of a variable length n." and "The Neural GPU architecture is the combination of a convolution on variable size inputs with a recurrent neural network") (Section 3 Model).

### Task: Binary multi-digit multiplication
- "The Neural GPU is a recent model that can learn algorithms such as multi-digit binary addition and binary multiplication in a way that generalizes to inputs of arbitrary length." (Abstract)
- "We observe similar generalization issues with multiplication." (Section 5 Generalization)
- Inference: In/Out Dimension are 1D (t), and Attention/State dynamics are Static/Constructed based on the variable-length sequence framing and recurrent convolutional architecture ("The Neural GPU consumes an input of a variable length n." and "The Neural GPU architecture is the combination of a convolution on variable size inputs with a recurrent neural network") (Section 3 Model).

### Task: Decimal arithmetic operations (all)
- "we have been able to learn to perform all the arithmetic operations (and generalize to arbitrarily long numbers) when the arguments are given in the decimal representation" (Abstract)
- "Figure 3: Training (top) and test error (bottom) on the decimal multiplication task." (Figure 3 caption)
- "We trained a decimal addition model with 1121 different seeds, and measure for each model the number of carries at which the error rate crosses 50%." (Section 5 Generalization)
- Inference: In/Out Dimension are 1D (t), and Attention/State dynamics are Static/Constructed based on the variable-length sequence framing and recurrent convolutional architecture ("The Neural GPU consumes an input of a variable length n." and "The Neural GPU architecture is the combination of a convolution on variable size inputs with a recurrent neural network") (Section 3 Model).

### Task: Binary multi-operand multiplication (3 terms)
- "We look at the binary multiplication of 3 terms." (Section 4 Improvements to the Neural GPU)
- "Figure 4: Influence of curriculum on 3-numbers multiplication task." (Figure 4 caption)
- Inference: In/Out Dimension are 1D (t) and In/Out Dynamics are Open (inferred) based on variable-length input descriptions and expression-length framing ("The Neural GPU consumes an input of a variable length n." and references to expression length) plus the recurrent convolutional architecture for Attention/State dynamics (Section 3 Model; Section 4 Improvements to the Neural GPU).

### Task: Binary arithmetic expression evaluation
- "We have also been able to train the Neural GPU to evaluate long arithmetic expressions with multiple operands that require respecting the precedence order of the operands, although these have succeeded only in their binary representation, and not with perfect accuracy." (Abstract)
- "Figure 5: Task of learning binary arithmetic on multiple numbers simultaneously using the operators  $+, -, \times, \div$ ." (Figure 5 caption)
- "Another experiment is to train a model on sequences of arithmetic operations with multiple numbers." (Section 4 Improvements to the Neural GPU)
- Inference: In/Out Dimension are 1D (t) and In/Out Dynamics are Open (inferred) based on variable-length input descriptions and expression-length framing ("The Neural GPU consumes an input of a variable length n." and "expressions of length 41"/"expressions of length 201") plus the recurrent convolutional architecture for Attention/State dynamics (Section 3 Model; Section 4 Improvements to the Neural GPU).
