# Rethinking Attention with Performers (2020)
Source: Rethinking Attention with Performers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly defines Performers as Transformer architectures and centers the method on approximating full-rank softmax attention.
- Auxiliary analyses consistently characterize the evaluated models as Transformer-style attention models across text, protein, and image tasks.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract and auxiliary evidence is already decisive.

## Evidence
- "We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers..." (Rethinking Attention with Performers.md:7, Abstract)
- "Positive (POS) softmax features (with redrawing) become crucial for achieving performance matching regular Transformers" (TASK_MODEL_RATIO.md:7, quote from Section 4.3)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision from the abstract and available auxiliary files; extending-dimensions file was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
