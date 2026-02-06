# Adaptive Computation Time for Recurrent Neural Networks (Not specified in the paper.)
Source: Adaptive Computation Time (ACT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| determining the parity of binary vectors | binary vectors (64 elements) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | binary target (odd/even parity) | 0D (inferred) | Fixed (inferred) |
| applying binary logic operations | input vectors with two bits and logic-gate chunks (sequence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | binary target sequence (truth values) | 1D (t) (inferred) | Capped (inferred) |
| adding integers | digit-encoded number vectors (sequence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | cumulative-sum digits (6 simultaneous classifications) | 1D (t) (inferred) | Capped (inferred) |
| sorting real numbers | real numbers with end-of-sequence flag (sequence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | sequence of indices in sorted order | 1D (t) (inferred) | Capped (inferred) |
| character prediction (Wikipedia) | byte sequence (characters) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | next-byte predictions | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper evaluates ACT on four synthetic supervised tasks—parity of binary vectors, binary logic operations, integer addition, and sorting real numbers—and on Wikipedia character prediction. Inputs are vectors or 1D sequences of bits, logic-gate vectors, digit encodings, real numbers, and bytes, with fixed or capped lengths, and outputs are binary labels or 1D sequences of digits/indices/bytes. Based on the recurrent formulation in Section 2, attention is treated as static and state as constructed (inferred).

## Evidence
### Task: determining the parity of binary vectors
- "presenting large binary vectors to the network and asking it to determine the parity." (Section 3.1 Parity)
- "The input vectors had 64 elements" (Section 3.1 Parity)
- "The corresponding target was 1 if there was an odd number of ones and 0 if there was an even number of ones." (Section 3.1 Parity)
- Inference: Mapped vector size to 1D (t) and Fixed dynamics; attention marked Static and state Constructed based on "computes the state sequence" and "The state is a fixed-size vector of real numbers containing the complete dynamic information of the network." (Section 2 Adaptive Computation Time)

### Task: applying binary logic operations
- "Each input sequence consists of a random number from 1 to 10 of size 102 input vectors." (Section 3.2 Logic)
- "The first two elements of each input represent a pair of binary numbers" (Section 3.2 Logic)
- "The binary target b_{B+1} for each input is the truth value yielded by recursively applying the B binary gates" (Section 3.2 Logic)
- Inference: Mapped sequence length to 1D (t) and Capped dynamics; attention marked Static and state Constructed based on "computes the state sequence" and "The state is a fixed-size vector of real numbers containing the complete dynamic information of the network." (Section 2 Adaptive Computation Time)

### Task: adding integers
- "The addition task presents the network with a input sequence of 1 to 5 size 50 input vectors." (Section 3.3 Addition)
- "Each vector represents a D digit number, where D is drawn randomly from 1 to 5" (Section 3.3 Addition)
- "The required output is the cumulative sum of all inputs up to the current one" (Section 3.3 Addition)
- "represented as a set of 6 simultaneous classifications for the 6 possible digits in the sum." (Section 3.3 Addition)
- Inference: Mapped sequence length to 1D (t) and Capped dynamics; attention marked Static and state Constructed based on "computes the state sequence" and "The state is a fixed-size vector of real numbers containing the complete dynamic information of the network." (Section 2 Adaptive Computation Time)

### Task: sorting real numbers
- "The sort task requires the network to sort sequences of 2 to 15 numbers drawn from a standard normal distribution in ascending order." (Section 3.4 Sort)
- "the random numbers were presented one at a time as inputs" (Section 3.4 Sort)
- "the required output was the sequence of indices into the number sequence placed in sorted order" (Section 3.4 Sort)
- Inference: Mapped sequence length to 1D (t) and Capped dynamics; attention marked Static and state Constructed based on "computes the state sequence" and "The state is a fixed-size vector of real numbers containing the complete dynamic information of the network." (Section 2 Adaptive Computation Time)

### Task: character prediction (Wikipedia)
- "The Wikipedia task is character prediction on text drawn from the Hutter prize Wikipedia dataset" (Section 3.5 Wikipedia Character Prediction)
- "one byte presented per input timestep and the next byte predicted as a target." (Section 3.5 Wikipedia Character Prediction)
- "Sequences of 500 consecutive bytes were randomly chosen from the training set" (Section 3.5 Wikipedia Character Prediction)
- Inference: Mapped fixed-length byte sequences to 1D (t) and Fixed dynamics; attention marked Static and state Constructed based on "computes the state sequence" and "The state is a fixed-size vector of real numbers containing the complete dynamic information of the network." (Section 2 Adaptive Computation Time)
