# Guiding High-Performance SAT Solvers with Unsat-Core Predictions (Not specified in the paper.)
Source: Guiding High-Performance SAT Solvers with Unsat-Core Predictions.md

## Core reasons
- Proposes a hybrid reasoning mechanism that injects neural unsat-core predictions into CDCL SAT solvers to guide branching decisions.
- The main contribution is a periodic refocusing algorithm that replaces solver variable activity scores with NeuroCore outputs, changing how computation proceeds.

## Evidence extracts
- "We modify several highperformance SAT solvers to periodically replace their variable activity scores with NeuroSAT's prediction of how likely the variables are to appear in an unsatisfiable core." (Abstract)
- "we settle for querying periodically on the entire problem (i.e. not conditioning on the trail) and replacing the variable activity scores with NeuroCore's prediction." (Section 4 Hybrid Solving: Extending CDCL with NeuroCore)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
