# Goal-Aware Neural SAT Solver (QuerySAT - goal-aware guidance) (Year not specified)
Source: Goal-Aware Neural SAT Solver (QuerySAT - goal-aware guidance).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes QuerySAT as a neural SAT solver built around a query mechanism and unsupervised loss, with no Transformer or self-attention architecture indicated.
- The auxiliary analyses describe CNF graph adjacency inputs and constructed recurrent state updates, which match GNN/MLP-style processing rather than Transformer blocks.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but the available abstract and auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "We then propose a neural SAT solver with a query mechanism called QuerySAT and show that it outperforms the neural baseline on a wide range of SAT tasks." (Goal-Aware Neural SAT Solver (QuerySAT - goal-aware guidance).md, Abstract line 7)
- "It receives a CNF Boolean formula φ in the input represented as two adjacency matrices." (TASK-DOMAINS.md line 15, cites Section IV-A)
- "At the beginning an empty state vector initialized with all ones is allocated for each variable and each clause" (TASK-DOMAINS.md line 19, cites Section IV-A)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Reviewed the abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; Extending-dimensions file was unavailable (`MISSING`); evidence was sufficient to decide.
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already conclusive.
