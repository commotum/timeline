# Enhancing Modern SAT Solver With Machine Learning Method (2025)
Source: Enhancing Modern SAT Solver With Machine Learning.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analyses consistently describe the central method as GNN-based (specifically WGCN + MLP) for SAT/UNSAT variable classification, not a Transformer/self-attention architecture.
- The extending-dimensions analysis input was unavailable (`MISSING`), but Pass 1 evidence from the abstract plus all available auxiliary files is sufficient for a high-confidence decision.

## Evidence
- "In this paper, we present a GNN-based algorithm that predicts at the same time backbone variables for SAT instances and UNSAT-core variables for UNSAT instances." (Enhancing Modern SAT Solver With Machine Learning.md, Abstract, line 44)
- "#### **Keywords**" / "SAT Solver, CDCL, Machine Learning, GNN" (Enhancing Modern SAT Solver With Machine Learning.md, Keywords, lines 46-48)
- "The paper uses GNNs to classify variables as backbone variables for SAT instances and as UNSAT-core variables for UNSAT instances" (TASK-DOMAINS.md, Summary, line 11)
- "The two GNN models designed for classification tasks are trained on distinct datasets to ensure specialized performance in different problem domains." (TASK_MODEL_RATIO.md, line 8)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-NO from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture identification.
