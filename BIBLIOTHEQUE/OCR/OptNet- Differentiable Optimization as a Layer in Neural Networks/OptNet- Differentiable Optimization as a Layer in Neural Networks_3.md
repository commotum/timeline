# OptNet: Differentiable Optimization as a Layer in Neural Networks (2017)
Source: OptNet- Differentiable Optimization as a Layer in Neural Networks.md

## Core reasons
- Proposes a new computation mechanism where a layer outputs the solution to a constrained optimization problem, enabling hard constraints and richer inference than standard feedforward layers.
- Provides differentiable argmin layers with implicit differentiation and a specialized QP solver so the optimization layer can be trained end-to-end, changing how computation happens inside the network.

## Evidence extracts
- "This paper presents OptNet, a network architecture that integrates optimization problems (here, specifically in the form of quadratic programs) as individual layers in larger end-to-end trainable deep networks." (Abstract)
- "In this paper, we consider how to treat exact, constrained optimization as an individual layer within a deep learning architecture." (1. Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
