# Generating Long Sequences with Sparse Transformers (Year not specified)
Source: Generating Long Sequences with Sparse Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly defines the method as a Transformer variant built around sparse factorizations of self-attention.
- Auxiliary analyses consistently describe Sparse Transformers as the trained architecture; the Extending-dimensions analysis file was unavailable (`MISSING`) but not needed for a high-confidence decision.

## Evidence
- "Transformers are powerful sequence models, but require time and memory that grows quadratically with the sequence length." (Generating Long Sequences with Sparse Transformers.md, Abstract)
- "We call networks with these changes Sparse Transformers" (Generating Long Sequences with Sparse Transformers.md, Abstract)
- "we restricted our investigation to a class of sparse attention patterns that have connectivity between all positions over several steps of attention." (TASK-DOMAINS.md, Evidence citing Section 4.1)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-YES from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; Extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient.
