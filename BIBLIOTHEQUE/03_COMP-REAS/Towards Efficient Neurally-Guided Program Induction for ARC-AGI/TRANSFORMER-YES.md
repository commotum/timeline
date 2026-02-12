# Towards Efficient Neurally-Guided Program Induction for ARC-AGI (2024)
Source: Towards Efficient Neurally-Guided Program Induction for ARC-AGI.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The method retained for submission (Learning the Program Space / GridCoder) is explicitly a transformer-based autoregressive program generator.
- Auxiliary analysis also shows a transformer encoder model in the LGS paradigm, reinforcing that self-attention architectures are materially used in the paper’s core modeling.
- The extending-dimensions analysis file was unavailable (`MISSING`), but available abstract + auxiliary evidence is already decisive.

## Evidence
- "the solution consists of training a transformer to output a program, using a pre-determined grammar (DSL) and syntax, that solves the task." (TASK-DOMAINS.md, Evidence -> Task: Program induction)
- "The experiments reported here used a Transformer encoder-only model with max pooling that outputs a flattened vector: a grid embedding." (TASK-DOMAINS.md, Evidence -> Task: Similarity prediction for execution-guided search)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer classification from the abstract plus TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient to finalize.
