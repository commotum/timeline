# GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS (Year not specified)
Source: Grokking- Generalization Beyond Overfitting on Small Algorithmic Datasets.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The auxiliary task/domain analysis directly identifies the trained architecture as a decoder-only Transformer with causal attention masking, which is core self-attention usage.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract + auxiliary evidence is already explicit enough for a high-confidence decision.

## Evidence
- "We trained a standard decoder-only transformer Vaswani et al. (2017) with causal attention masking, and calculated loss and accuracy only on the answer part of the equation." (TASK-DOMAINS.md, Evidence section citing Appendix A.1.2)
- "Attention and state dynamics are inferred from the fixed equation format and the decoder-only transformer with causal attention masking." (TASK-DOMAINS.md, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract plus auxiliary analysis; central model is explicitly described as decoder-only Transformer with causal attention.
Pass 2 (targeted source scan): skipped - Not needed because Pass 1 already provided explicit architecture-level Transformer evidence.
