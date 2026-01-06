# Proximal Policy Optimization Algorithms (Not specified in the paper.)
Source: Proximal Policy Optimization (PPO).md

## Core reasons
- The paper proposes new policy gradient methods and an optimization objective for reinforcement learning, which is a training/optimization contribution rather than positional encoding or dimensional adaptation.
- The core contribution is a clipped surrogate objective and first-order optimization procedure to improve stability and sample efficiency, aligning with ML foundations and principles.

## Evidence extracts
- "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a \"surrogate\" objective function using stochastic gradient ascent." (Abstract)
- "We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy." (Section 1 Introduction)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
