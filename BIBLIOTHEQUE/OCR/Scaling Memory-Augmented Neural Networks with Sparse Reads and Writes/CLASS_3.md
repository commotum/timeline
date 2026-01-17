# Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes (Not specified in the paper.)
Source: Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes.md

## Core reasons
- Proposes a new memory access mechanism (SAM) to change how computation and memory access happen in memory-augmented neural networks, addressing scaling limits.
- The contribution centers on sparse read/write operations and efficient data structures to enable large external memory with end-to-end differentiable training.

## Evidence extracts
- "Here, we present an end-to-end differentiable memory access scheme, which we call Sparse Access Memory (SAM), that retains the representational power of the original approaches whilst training efficiently with very large memories." (Abstract)
- "This paper introduces *Sparse Access Memory (SAM)*, a new neural memory architecture with two innovations. Most importantly, all writes to and reads from external memory are constrained to a sparse subset of the memory words, providing similar functionality as the NTM, while allowing computational and memory efficient operation." (Section 3 Architecture)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
