# Learning Semantic String Transformations from Examples (2012)
Source: Learning Semantic String Transformations from Examples.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes a symbolic programming-by-example method based on transformation languages, table lookup, and synthesis, not a neural architecture with self-attention.
- Auxiliary analyses provide no Transformer/attention-model signals; attention/state dynamics are marked as not specified, and the implementation is described as an inductive synthesis add-in.

## Evidence
- "We describe an expressive transformation language for semantic manipulation that combines table lookup operations and syntactic manipulations." (Abstract, Learning Semantic String Transformations from Examples.md)
- "Attention Dynamic | Not specified in the paper." (Task Table, TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO decision; no central self-attention/Transformer model indicated. Extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 already sufficient.
