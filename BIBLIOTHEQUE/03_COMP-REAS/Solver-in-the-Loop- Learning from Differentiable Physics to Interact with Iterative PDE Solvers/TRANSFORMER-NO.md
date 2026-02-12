# Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers (Year not specified)
Source: Solver-in-the-Loop- Learning from Differentiable Physics to Interact with Iterative PDE Solvers.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analyses describe solver-in-the-loop differentiable physics with learned correction networks, not Transformer/self-attention architectures.
- Available auxiliary files indicate a fully convolutional model family and static/direct mapping; the Extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "We find that previously used learning approaches are significantly outperformed by methods that integrate the solver into the training loop and thereby allow the model to interact with the PDE during training." (Abstract, `Solver-in-the-Loop- Learning from Differentiable Physics to Interact with Iterative PDE Solvers.md`)
- "The neural network component  $F(s \mid \theta)$  of the correction function is realized with a fully convolutional architecture." (Quoted in `TASK-DOMAINS.md`, Evidence section, from Section 3.2 Training Procedure)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence non-Transformer classification.
Pass 2 (targeted source scan): skipped - Pass 1 already established the central model family without self-attention signals.
