# GROKKING MODULAR POLYNOMIALS (Not specified in the paper.)
Source: Grokking Modular Polynomials.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Modular addition (many terms) computation | Modular integers n1...nS (one_hot vectors, concatenated) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Modular sum label in Z_p (one_hot vector) | 0D (inferred) | Fixed (inferred) |
| Modular multiplication (two-variable) computation | Modular integers n1, n2 (one_hot vectors, concatenated) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Modular product label in Z_p (one_hot vector) | 0D (inferred) | Fixed (inferred) |
| Arbitrary modular polynomial (two-variable) computation | Modular integers n1, n2 (one_hot vectors, concatenated) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Modular polynomial value in Z_p (one_hot vector) (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper studies MLPs that compute modular arithmetic functions: multi-term modular addition, modular multiplication, and general modular polynomials over two variables. Inputs are described as concatenated one_hot encodings of modular integers, and outputs are modular class labels in Z_p, yielding inferred 1D fixed-length inputs and 0D fixed outputs. The architectures are feed-forward MLPs with fixed input interfaces, so attention is static and state is direct (inferred).

## Evidence
### Task: Modular addition (many terms) computation
- "Consider the modular addition task with many terms with arbitrary coefficients:" (Section 2)
- "$e_{n_1},\ldots,e_{n_S}\in\mathbb{R}^p$ are one_hot encoded numbers" (Section 2)
- "The targets are one_hot encoded answers" (Section 2)
- Inference: Input/output dimensions, dynamics, attention, and state are inferred from the fixed concatenation in "$\mathbf{f}_{addS}(\mathbf{e}_{n_1} \oplus \cdots \oplus \mathbf{e}_{n_S})$" and the use of a 2-layer MLP (Section 2).

### Task: Modular multiplication (two-variable) computation
- "Now consider the modular multiplication task in two variables:" (Section 3)
- "f_{mul2}(e_{n_1} \oplus e_{n_2})" (Section 3)
- "desired one_hot encoded labels for the modular multiplication task" (Appendix A.1)
- Inference: Input/output dimensions, dynamics, attention, and state are inferred from the fixed two-input MLP form in equation (5) and the 2-layer MLP setup (Section 3).

### Task: Arbitrary modular polynomial (two-variable) computation
- "Consider a general modular polynomial in two variables $(n_1, n_2)$ containing S terms:" (Section 4)
- "input to the network are stacked, one_hot represented numbers $n_1, n_2$" (Section 4)
- "z = f_{addS} \left( u^{(1)} \oplus \cdots \oplus u^{(S)} \right)" (Section 4)
- Inference: Output as a one_hot modular class label and all dimension/dynamics/attention/state labels are inferred from the reuse of $f_{addS}$ plus the fixed, concatenated input interface (Sections 2 and 4).

## CSV Output (required)
