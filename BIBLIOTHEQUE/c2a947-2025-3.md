# Enhancing Modern SAT Solver With Machine Learning Method (2025)
Source: c2a947-2025.pdf

## Core reasons
- Proposes a GNN-based algorithm to predict backbone and UNSAT-core variables to guide SAT solving, which is a mechanism for reasoning over SAT instances.
- Integrates neural predictions into CDCL decision queues/scores to alter branching and search behavior, changing how the solver computes solutions.

## Evidence extracts
- "a GNN-based algorithm that predicts at the same time backbone variables for SAT instances and UNSAT-core variables for UNSAT instances." (p. 2)
- "The GNN generates probabilities indicating the likelihood of variables being part of the backbone or the UNSAT-core. These probabilities are then assigned to the CDCL solver to initialize the variable decision queue and decision scores, guiding the solving process more effectively." (p. 4)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
