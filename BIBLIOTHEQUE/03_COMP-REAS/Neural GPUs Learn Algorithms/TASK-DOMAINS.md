# NEURAL GPUS LEARN ALGORITHMS (Not specified in the paper)
Source: Neural GPUs Learn Algorithms.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Long binary addition | Binary digit sequence for two lower-endian numbers with a separator symbol (PAD possible) | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | Binary digit sequence for the sum (lower-endian) | 1D (t) | Open (inferred) |
| Long binary multiplication | Binary digit sequence for two lower-endian numbers with a separator symbol (PAD possible) | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | Binary digit sequence for the product (lower-endian) | 1D (t) | Open (inferred) |
| Copying sequences | Bit sequence | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | Bit sequence identical to input | 1D (t) | Open (inferred) |
| Reversing sequences | Bit sequence | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | Bit sequence in reverse order | 1D (t) | Open (inferred) |
| Duplicating sequences | Bit sequence (with padding to match output length) | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | Bit sequence duplicated twice | 1D (t) | Open (inferred) |
| Counting by sorting bits | Bit sequence | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | Bit sequence sorted as all 0s then 1s | 1D (t) | Open (inferred) |

## Summary
The paper evaluates Neural GPUs on algorithmic sequence-transduction tasks over binary symbol sequences, including arithmetic (addition and multiplication) and simpler sequence transformations (copy, reverse, duplicate, and bit-sorting/counting). All tasks operate on 1D sequences and produce 1D sequences. The text emphasizes handling inputs of arbitrary size, so input/output dynamics are treated as open, while the model processes fixed inputs without runtime attention selection and evolves an internal recurrent state (static attention and constructed state, inferred from the architecture description).

## Evidence
### Task: Long binary addition
- "The two core tasks on which we study the performance of Neural GPUs are long binary addition and long binary multiplication." (Section 3.1)
- "**Long binary addition (badd)** is the task of adding two numbers represented lower-endian in binary notation." (Section 3.1)
- "As described in Section 2, we input a sequence of discrete symbols into the network and we read out a sequence of symbols again." (Section 3.1)
- Inference: 1D (t) and open dynamics are inferred because the task is defined over sequences and the paper stresses "inputs of arbitrary size" and generalization to longer sequences; attention is static and state is constructed because "all inputs are written into the starting state" and "This mental image evolves in time" via recurrent updates. (Abstract; Section 2)

### Task: Long binary multiplication
- "The two core tasks on which we study the performance of Neural GPUs are long binary addition and long binary multiplication." (Section 3.1)
- "**Long binary multiplication (bmu1)** is the task of multiplying two binary numbers, represented lower-endian." (Section 3.1)
- "As described in Section 2, we input a sequence of discrete symbols into the network and we read out a sequence of symbols again." (Section 3.1)
- Inference: 1D (t) and open dynamics are inferred because the task is defined over sequences and the paper stresses "inputs of arbitrary size" and generalization to longer sequences; attention is static and state is constructed because "all inputs are written into the starting state" and "This mental image evolves in time" via recurrent updates. (Abstract; Section 2)

### Task: Copying sequences
- "**Copying sequences** is the simple task of producing on output the same sequence as on input." (Section 3.2)
- Inference: 1D (t) and open dynamics are inferred from the sequence framing and the claim that tasks generalize to longer lengths; attention is static and state is constructed based on the architecture where "all inputs are written into the starting state" and "This mental image evolves in time". (Section 3.2; Section 2)

### Task: Reversing sequences
- "**Reversing sequences** is the task of reversing a sequence of bits, n is the length of the sequence." (Section 3.2)
- Inference: 1D (t) and open dynamics are inferred from the sequence framing and the claim that tasks generalize to longer lengths; attention is static and state is constructed based on the architecture where "all inputs are written into the starting state" and "This mental image evolves in time". (Section 3.2; Section 2)

### Task: Duplicating sequences
- "**Duplicating sequences** is the task of duplicating the input bit sequence on output twice, as in the example below." (Section 3.2)
- "We use the padding symbol on input to make it match the output length." (Section 3.2)
- Inference: 1D (t) and open dynamics are inferred from the sequence framing and the claim that tasks generalize to longer lengths; attention is static and state is constructed based on the architecture where "all inputs are written into the starting state" and "This mental image evolves in time". (Section 3.2; Section 2)

### Task: Counting by sorting bits
- "**Counting by sorting bits** is the task of sorting the input bit sequence on output." (Section 3.2)
- "the network must count how many 0s are in the input" (Section 3.2)
- Inference: 1D (t) and open dynamics are inferred from the sequence framing and the claim that tasks generalize to longer lengths; attention is static and state is constructed based on the architecture where "all inputs are written into the starting state" and "This mental image evolves in time". (Section 3.2; Section 2)
