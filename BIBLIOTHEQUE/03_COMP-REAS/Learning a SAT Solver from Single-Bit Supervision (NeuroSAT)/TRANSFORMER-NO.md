# Learning a SAT Solver from Single-Bit Super-Vision (Year not specified)
Source: Learning a SAT Solver from Single-Bit Supervision (NeuroSAT).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract identifies NeuroSAT as a "message passing neural network," which is not a Transformer-style self-attention architecture.
- The auxiliary analyses describe iterative graph message passing as the core mechanism and do not indicate Transformer-family blocks as central to the main results.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "We present NeuroSAT, a message passing neural network that learns to solve SAT problems after only being trained as a classifier to predict satisfiability." (Abstract, `Learning a SAT Solver from Single-Bit Supervision (NeuroSAT).md`)
- "Based on the described graph/adjacency-matrix input and iterative message passing, the tasks operate over variable-size inputs with constructed state and static attention..." (`TASK-DOMAINS.md`, Summary)
- "Inference: In Dimension 2D (x, y) and In Dynamics Open are inferred because the input is \"any bipartite adjacency matrix M over any number of literals and clauses\"; Attention/State are inferred from \"iteratively refines a vector space embedding for each node by passing \"messages\" back and forth\" (Section 3 Model)." (`TASK-DOMAINS.md`, Evidence)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-NO decision.
Pass 2 (targeted source scan): skipped - pass 1 already provided clear architecture evidence.
