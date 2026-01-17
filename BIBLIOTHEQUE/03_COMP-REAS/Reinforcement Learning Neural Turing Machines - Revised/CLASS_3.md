# Reinforcement Learning Neural Turing Machines - Revised (Not specified in the paper.)
Source: Reinforcement Learning Neural Turing Machines - Revised.md

## Core reasons
- Proposes a neural controller that learns to interact with discrete external interfaces (input, memory, output tapes), addressing the need for external memory and discrete actions beyond standard feedforward computation.
- The central contribution is a computation mechanism using Reinforce to train discrete action decisions for memory/tape access, not a dataset or positional encoding change.

## Evidence extracts
- "We examine feasibility of learning models to interact with discrete Interfaces. We investigate the following discrete Interfaces: a memory Tape, an input Tape, and an output Tape. We use a Reinforcement Learning algorithm to train a neural network that interacts with such Interfaces to solve simple algorithmic tasks." (Abstract)
- "Our concrete proposal is to use the Reinforce algorithm to learn *where* to access the discrete interfaces, and to use the backpropagation algorithm to determine *what* to write to the memory and to the output. We call this model the RL–NTM." (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
