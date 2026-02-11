# Adaptive Attention Span in Transformers (Year not specified)
Source: TASK-DOMAINS.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: hint-only

## Why
- The hint evidence explicitly describes per-head attention span masking, which is a self-attention mechanism at the core of the model.
- The work centers on an adaptive attention mechanism in a Transformer context rather than mentioning Transformers only as a baseline.

## Evidence
- "For each head, we add a masking function to control for the span of the attention." (TASK-DOMAINS.md, Evidence section quoting Section 2.2)
- "In this section, we evaluate the impact of our adaptive attention mechanism in the experimental setting of Al-Rfou et al. (2019) for character level language modeling." (TASK_MODEL_RATIO.md, item 1 quoting Section 3, Experiments)

## Pass accounting
Pass 0 (hint-first): performed - sufficient direct evidence of core self-attention usage in the model.
Pass 1 (source triage): skipped - hint evidence was sufficient for a high-confidence decision.
Pass 2 (source deep dive): skipped - no remaining ambiguity after Pass 0.
