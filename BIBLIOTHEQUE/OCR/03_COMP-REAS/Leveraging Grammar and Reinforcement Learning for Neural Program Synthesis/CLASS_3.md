# Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis (Not specified in the paper.)
Source: Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis.md

## Core reasons
- Proposes a reinforcement-learning objective to optimize for generating any consistent program, changing how neural program synthesis computation is trained beyond standard feedforward MLE.
- Introduces syntax/grammar-based conditioning to prune invalid programs, adding a mechanism that constrains program generation rather than providing data or positional encodings.

## Evidence extracts
- "To address this problem, we alter the optimization objective: instead of maximum likelihood, we use policy gradient reinforcement learning to directly encourage generation of *any* program that is consistent with the given examples." (Section 1 Introduction)
- "Similarly to the work of Parisotto et al. (2017), we explore a method for leveraging the syntax of the programming language in order to aggressively prune the exponentially large search space of possible programs." (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
