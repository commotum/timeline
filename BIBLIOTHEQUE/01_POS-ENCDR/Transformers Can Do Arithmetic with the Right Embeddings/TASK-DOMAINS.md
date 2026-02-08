# Transformers Can Do Arithmetic with the Right Embeddings (Year not specified in the paper)
Source: Transformers Can Do Arithmetic with the Right Embeddings.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Addition | Reversed integer character-token sequence with two operands and arithmetic symbols | 1D (t) (inferred) | Capped | Static (inferred) | Direct (inferred) | Character-token sequence of the sum | 1D (t) (inferred) | Capped |
| Subtraction | Reversed integer character-token sequence with two operands and arithmetic symbols (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Character-token sequence of the difference (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Multiplication | Reversed integer character-token sequence with two operands and arithmetic symbols (inferred) | 1D (t) (inferred) | Capped | Static (inferred) | Direct (inferred) | Character-token sequence of the product | 1D (t) (inferred) | Capped |
| Array sorting | Array of indexed reversed integers | 2D (x, y) (inferred) | Capped | Static (inferred) | Direct (inferred) | Character-index sequence in ascending numeric order | 1D (t) (inferred) | Capped |
| Bitwise OR | Two left-aligned binary vectors | 1D (t) (inferred) | Capped | Static (inferred) | Direct (inferred) | Left-aligned position-wise OR binary vector | 1D (t) (inferred) | Capped |

## Summary
The paper evaluates five algorithmic reasoning tasks framed as sequence generation: addition, subtraction, multiplication, array sorting, and bitwise OR. The supported task domains are primarily 1D (t) symbolic sequences, with sorting using a two-axis structured input (2D (x, y), inferred from the paper’s separate array-length and number-length axes). All tasks are Capped by explicit maximum operand/vector/array sizes and by context-length limits. From the described decoder-only autoregressive setup, attention is Static (inferred) and state is Direct (inferred) across tasks.

## Evidence
### Task: Addition
- "We train decoder-only causal language models to solve addition problems." (Section 3, Experimental Setup)
- "inputs are formatted least significant digit first, e.g. 98282 + 3859172 = 2787472." (Section 3, Experimental Setup)
- "We train on all combinations of operand lengths less than or equal to i and j where i and j are the maximum lengths of the first and second operands, respectively." (Section 3, Experimental Setup)
- Inference: `1D (t)`, `Static`, and `Direct` are inferred from sequence-formatted character-token arithmetic and "standard autoregressive transformer" training; `Capped` is supported by explicit max operand lengths and "capped by the context length." (Section 3, Experimental Setup; Section 5)

### Task: Subtraction
- "We train models on a dataset made up of an even mix of addition and subtraction samples." (Section 4.1 Addition and Subtraction)
- "these small transformer models can simultaneously learn to extrapolate for both the symmetric operation of addition and the anti-symmetric operation of subtraction" (Section 4.1 Addition and Subtraction)
- "trained with exactly the same hyperparameters used to train the addition models above." (Section 4.1 Addition and Subtraction)
- Inference: Input/output form, `1D (t)`, and `Capped` are inferred from the explicit reuse of the addition setup; `Static` and `Direct` are inferred from the same autoregressive transformer setup used across experiments. (Section 4.1; Section 3, Experimental Setup)

### Task: Multiplication
- "We now study a harder task, multiplication of natural numbers, where the length of the output may be the sum of the lengths of the operands." (Section 4.2 Integer Multiplication)
- "Multiplication: We implement the multiplication datasets for both training and testing the exact same manor as for addition, only changing the operation used to calculate the answer." (Appendix A.2 Datasets)
- "The red square denotes in distribution testing on up to 15 digit operands." (Figure 6 caption, Section 4.2 Integer Multiplication)
- Inference: `1D (t)`, `Static`, and `Direct` are inferred from reuse of the addition-style serialized sequence setup and shared transformer formulation; `Capped` is supported by explicit operand-size limits. (Section 4.2; Appendix A.2; Figure 6 caption)

### Task: Array sorting
- "we now analyze the task of sorting arrays of multiple variable length numbers" (Section 4.3 Array Sorting)
- "We present each sorting problem using alphabetical indices for each (reversed) number in an input array where the expected output is the alphabetical indices in ascending order." (Section 4.3 Array Sorting)
- "We train with arrays of up to 10 numbers each having up to 10 digits and then evaluate with arrays of up to 30 numbers each having up to 30 digits." (Section 4.3 Array Sorting)
- Inference: `2D (x, y)` is inferred because the paper explicitly evaluates two structural axes (number length and array length); output remains `1D (t)` as an index sequence; `Static` and `Direct` are inferred from the same autoregressive transformer setup. (Section 4.3; Section 3, Experimental Setup)

### Task: Bitwise OR
- "The input for this problem is two binary vectors, the longer input vector is all zeros and the shorter input contains a one." (Appendix A.2 Datasets)
- "The output should be the length of the longer vector with the one in the same position as in the shorter vector." (Appendix A.2 Datasets)
- "we analyze the bitwise OR task, where the model has to output left aligned position wise OR of two binary vectors." (Appendix A.3 Bitwise OR on Binary Vectors)
- "the maximum length of either input vector is twenty." (Appendix A.3 Bitwise OR on Binary Vectors)
- Inference: `1D (t)` is inferred from position-wise vector indexing; `Static` and `Direct` are inferred from the shared transformer setup; `Capped` is explicitly supported by the max input-vector length. (Appendix A.2; Appendix A.3; Section 3, Experimental Setup)
