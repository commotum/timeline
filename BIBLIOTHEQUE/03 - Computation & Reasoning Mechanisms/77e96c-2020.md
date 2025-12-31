# Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers (2020)
Source: 77e96c-2020.pdf

## Core reasons
- Proposes a solver-in-the-loop training setup that integrates a learned correction model into iterative PDE solvers via differentiable physics, changing how computation/training proceeds through recurrent solver interaction.
- Focuses on an interactive computation mechanism for reducing numerical errors in iterative solvers, not on datasets, benchmarks, or positional encoding changes.

## Evidence extracts
- "We ﬁnd that previously used learning approaches are signiﬁcantly outperformed by methods that integrate the solver into the training loop and thereby allow the model to interact with the PDE during training." (p. 1)
- "The core of most numerical methods contains some form of iterative process – either in the form of repeated updates over time for explicit solvers or even within a single update step for implicit solvers. Hence, we focus on iterative PDE solving algorithms [17]." (p. 2)
- "Solver-in-the-loop (SOL): By integrating the learned function into a differentiable physics pipeline, the corrections can interact with the physical system, alter the states, and receive gradients about the future performance of these modiﬁcations." (p. 4)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
