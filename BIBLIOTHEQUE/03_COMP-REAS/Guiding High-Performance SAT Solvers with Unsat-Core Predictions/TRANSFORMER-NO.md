# Guiding High-Performance SAT Solvers with Unsat-Core Predictions (Year not specified)
Source: Guiding High-Performance SAT Solvers with Unsat-Core Predictions.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analyses describe NeuroSAT/NeuroCore for SAT unsat-core prediction and CDCL heuristic guidance, with no Transformer/self-attention blocks as a core model component.
- Available auxiliary cues characterize the model as message passing plus MLP-style updates and static attention dynamics, which is not Transformer-style self-attention.
- Extending-dimensions analysis markdown was unavailable (`MISSING`), but the available evidence is still sufficient for a high-confidence decision.

## Evidence
- "we train a simplified NeuroSAT architecture to directly predict the unsatisfiable cores of real problems." (Guiding High-Performance SAT Solvers with Unsat-Core Predictions.md, Abstract)
- "the network performs T iterations of \"message passing\"" (TASK-DOMAINS.md, Evidence: Task unsat-core variable prediction)
- "unsat-core variable prediction,CNF clauses/literals (Boolean formula),\"2D (x, y) (inferred)\",Capped (inferred),Static (inferred),Constructed (inferred),variable scores / core-membership probabilities,1D (t) (inferred),Capped (inferred)" (TASK-DOMAINS.csv, row: unsat-core variable prediction)
- "Thus, fine-tuning the network is relatively unimportant, and we only ever trained with a single set of hyperparameters." (TASK_MODEL_RATIO.md, item 2 evidence)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for high-confidence TRANSFORMER-NO from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
