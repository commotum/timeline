# Trust Region Policy Optimization (2015)
Source: Trust Region Policy Optimization (TRPO).md

## Core reasons
- Proposes a new policy optimization algorithm (TRPO) with monotonic improvement guarantees, focusing on how to optimize policies rather than on data or model architecture.
- Formulates trust-region constrained updates using KL divergence to guide policy learning, framing the contribution as an optimization/training method.

## Evidence extracts
- "We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy Optimization (TRPO)." (Abstract)
- "One way to take larger steps in a robust way is to use a constraint on the KL divergence between the new policy and the old policy, i.e., a trust region constraint:" (Section 4 Optimization of Parameterized Policies)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
