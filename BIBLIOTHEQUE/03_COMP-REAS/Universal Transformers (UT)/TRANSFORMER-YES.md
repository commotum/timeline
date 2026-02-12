# Universal Transformers (Year not specified)
Source: Universal Transformers (UT).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly defines the proposed model as a "self-attentive recurrent sequence model" and a generalization of Transformer.
- Auxiliary analyses consistently characterize the evaluated model family as Universal Transformer with self-attention as a central mechanism; the Extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "We propose the Universal Transformer (UT), a parallel-in-time self-attentive recurrent sequence model which can be cast as a generalization of the Transformer model and which addresses these issues." (Universal Transformers (UT).md, Abstract, line 13)
- "Attention Dynamic is `Static` from fixed-sequence self-attention (\"using a self-attention mechanism to exchange information across all positions in the sequence,\" Section 2.1)" (TASK-DOMAINS.md, line 29)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence TRANSFORMER-YES decision using abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; Extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
