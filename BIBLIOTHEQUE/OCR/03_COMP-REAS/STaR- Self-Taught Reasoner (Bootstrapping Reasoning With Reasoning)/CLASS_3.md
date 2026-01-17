# STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning (Not specified in the paper.)
Source: STaR- Self-Taught Reasoner (Bootstrapping Reasoning With Reasoning).md

## Core reasons
- Proposes an iterative bootstrapping loop that changes how the model learns to reason by generating rationales, filtering by correctness, and fine-tuning on those rationales.
- Introduces a rationalization step that alters the computation/training process by generating rationales given the correct answer to improve reasoning coverage.

## Evidence extracts
- "We propose a technique to iteratively leverage a small number of rationale examples and a large dataset without rationales, to bootstrap the ability to perform successively more complex reasoning." (Abstract)
- "we propose **rationalization**: for each problem that the model fails to answer correctly, we generate a new rationale by providing the model with the correct answer." (Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
