# LIE-ACCESS NEURAL TURING MACHINES (Not specified in the paper.)
Source: Lie-Access Neural Turing Machines (LANTM).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sequence copy | Symbol sequence $a_1 a_2 a_3 \cdots a_k$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Symbol sequence $a_1a_2a_3\cdots a_k$ | 1D (t) | Capped |
| Sequence reversal | Symbol sequence $a_1a_2a_3\cdots a_k$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Symbol sequence $a_k a_{k-1} a_{k-2} \cdots a_1$ | 1D (t) | Capped |
| Bigram flip (pairwise swap) | Symbol sequence $a_1a_2a_3a_4\cdots a_{2k-1}a_{2k}$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Symbol sequence $a_2a_1a_4a_3\cdots a_{2k}a_{2k-1}$ | 1D (t) | Capped |
| Integer doubling | Digit sequence $a_1 a_2 \cdots a_k$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Digit sequence representing $2 \times a_k \cdots a_1$ | 1D (t) | Capped |
| Interleaved addition | Interleaved digit sequence $a_1a_2a_3a_4\cdots a_{2k-1}a_{2k}$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Digit sequence for $a_{2k}a_{2k-2}\cdots a_2 + a_{2k-1}\cdots a_1$ | 1D (t) | Capped |
| Odd-first reordering | Symbol sequence $a_1a_2a_3a_4\cdots a_{2k-1}a_{2k}$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Symbol sequence $a_1a_3\cdots a_{2k-1}a_2a_4\cdots a_{2k}$ | 1D (t) | Capped |
| Repeat copy | Symbol sequence $\overline{N}a_1\cdots a_{20}$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Symbol sequence $a_1 \cdots a_{20} \cdots a_1 \cdots a_{20}$ (N times) | 1D (t) | Capped |
| Priority sort | Unary-priority sequence $\overline{5}a_{5}\overline{2}a_{2}\overline{9}a_{9}\cdots$ | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | Symbol sequence $a_1a_2a_3\cdots a_k$ | 1D (t) | Capped |

## Summary
The paper evaluates LANTM on eight algorithmic sequence-transduction tasks (copying, reversal, pairwise swaps, arithmetic doubling and addition, odd/even reordering, repetition, and priority sorting). All tasks operate on 1D token/digit sequences with bounded lengths, so input and output dynamics are capped. The model uses memory heads and external memory, supporting dynamic attention and constructed state (both inferred from the described read/write mechanism).

## Evidence
### Task: Sequence copy
- "1 - COPY | $a_1 a_2 a_3 \cdots a_k$ | $a_1a_2a_3\cdots a_k$" (Section 5 Experiments, Table 1a)
- "Our experiments are on a series of algorithmic tasks shown in Table 1a." (Section 5 Experiments, Tasks)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)

### Task: Sequence reversal
- "2 - Reverse | $a_1a_2a_3\cdots a_k$ | $a_k a_{k-1} a_{k-2} \cdots a_1$" (Section 5 Experiments, Table 1a)
- "Our experiments are on a series of algorithmic tasks shown in Table 1a." (Section 5 Experiments, Tasks)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)

### Task: Bigram flip (pairwise swap)
- "3 - BIGRAM FLIP | $a_1a_2a_3a_4\cdots a_{2k-1}a_{2k}$ | $a_2a_1a_4a_3\cdots a_{2k}a_{2k-1}$" (Section 5 Experiments, Table 1a)
- "Our experiments are on a series of algorithmic tasks shown in Table 1a." (Section 5 Experiments, Tasks)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)

### Task: Integer doubling
- "4 - Double | $a_1 a_2 \cdots a_k$ | $2 \times  a_k \cdots a_1$" (Section 5 Experiments, Table 1a)
- "The DOUBLE task takes an integer $x\in[0,10^k)$ padded to k digits and outputs 2x in k+1 digits" (Section 5 Experiments, Task descriptions and parameters)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)

### Task: Interleaved addition
- "5 - Interleaved Add | $a_1a_2a_3a_4\cdots a_{2k-1}a_{2k}$ | $ a_{2k}a_{2k-2}\cdots a_2 + a_{2k-1}\cdots a_1 $" (Section 5 Experiments, Table 1a)
- "The Interleaved ADD task takes two integers $x,y\in[0,10^k)$ padded to k digits and interleaved" (Section 5 Experiments, Task descriptions and parameters)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)

### Task: Odd-first reordering
- "6 - Odd First | $a_1a_2a_3a_4\cdots a_{2k-1}a_{2k}$ | $a_1a_3\cdots a_{2k-1}a_2a_4\cdots a_{2k}$" (Section 5 Experiments, Table 1a)
- "In ODD FIRST, the model must output the odd-indexed elements first, followed by the even-indexed elements." (Section 5 Experiments, Tasks)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)

### Task: Repeat copy
- "7 - Repeat Copy | $\overline{N}a_1\cdots a_{20}$ | $a_1 \cdots a_{20} \cdots a_1 \cdots a_{20}$ (N times)" (Section 5 Experiments, Table 1a)
- "In REPEAT COPY, each model must repeat a sequence of length 20, N times." (Section 5 Experiments, Tasks)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)

### Task: Priority sort
- "8 - Priority Sort | $\overline{5}a_{5}\overline{2}a_{2}\overline{9}a_{9}\cdots$ | $a_1a_2a_3\cdots a_k$" (Section 5 Experiments, Table 1a)
- "In PRIORITY SORT, each item of the input sequence is given a priority, and the model must output them in priority order." (Section 5 Experiments, Tasks)
- Inference: Attention Dynamic and State Dynamic are inferred because the model "maintains a read head q" that is updated by actions and reads memory, and "a new memory is automatically appended to $\Sigma$." (Section 4.1 Addressing Procedure; Section 4.2 Reading and Writing Memories)
