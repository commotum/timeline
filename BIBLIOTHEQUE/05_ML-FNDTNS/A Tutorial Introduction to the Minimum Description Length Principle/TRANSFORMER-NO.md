# A Tutorial Introduction to the Minimum Description Length Principle (2004)
Source: A Tutorial Introduction to the Minimum Description Length Principle.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- The hints describe a methodological/statistical tutorial on the MDL principle for model selection and compression, not a neural architecture paper.
- The hint summary explicitly states that attention mechanisms are not specified, with no indication that self-attention is part of any central model.

## Evidence
- "The Minimum Description Length (MDL) Principle is a relatively recent method for inductive inference that provides a generic solution to the model selection problem." (TASK-DOMAINS.md, Section 1.1 Introduction and Overview)
- "The paper does not specify concrete input/output dimensionality, dynamics, attention, or state mechanisms for these tasks." (TASK-DOMAINS.md, Summary)

## Pass accounting
Pass 0 (hint-first): performed - sufficient evidence for a high-confidence non-Transformer classification from TASK-DOMAINS.md/TASK-DOMAINS.csv/TASK_MODEL_RATIO.md.
Pass 1 (source triage): skipped - hint evidence already decisive; no Transformer/self-attention cues present.
Pass 2 (source deep dive): skipped - not needed after decisive hint-first triage.
