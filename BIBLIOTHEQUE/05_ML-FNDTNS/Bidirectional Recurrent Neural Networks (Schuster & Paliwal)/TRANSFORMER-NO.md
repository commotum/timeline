# Bidirectional Recurrent Neural Networks (1997)
Source: Bidirectional Recurrent Neural Networks (Schuster & Paliwal).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s central method is explicitly a bidirectional recurrent neural network (BRNN), not a Transformer or self-attention architecture.
- Auxiliary analyses consistently describe recurrent/state-based modeling; `EXTENDING-DIMENSIONS.md` was unavailable and was skipped.

## Evidence
- "a regular recurrent neural network (RNN) is extended to a bidirectional recurrent neural network (BRNN)." (Abstract, `Bidirectional Recurrent Neural Networks (Schuster & Paliwal).md`)
- "Index Terms—Recurrent neural networks." (Abstract, `Bidirectional Recurrent Neural Networks (Schuster & Paliwal).md`)
- "The recurrent architectures use explicit state neurons" (Summary, `TASK-DOMAINS.md`)
- "Two different structures of the modified BRNN ... are trained separately as classifiers" (Item 2, `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for high-confidence TRANSFORMER-NO from abstract and auxiliary files; no Transformer/self-attention signals.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient to finalize.
