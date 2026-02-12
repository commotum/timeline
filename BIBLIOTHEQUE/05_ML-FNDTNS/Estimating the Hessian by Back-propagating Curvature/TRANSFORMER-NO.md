# Estimating the Hessian by Back-propagating Curvature (2012)
Source: Estimating the Hessian by Back-propagating Curvature.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analyses describe Curvature Propagation for Hessian estimation on computational graphs and score matching with cRBM/small neural networks, not Transformer/self-attention architectures.
- Auxiliary files explicitly indicate attention is not specified, and the extending-dimensions analysis file was unavailable (`MISSING`), with no contrary Transformer evidence in available Pass 1 materials.

## Evidence
- "In this work we develop Curvature Propagation (CP), a general technique for efficiently computing unbiased approximations of the Hessian of any function that is computed using a computational graph." (Estimating the Hessian by Back-propagating Curvature.md, Abstract, line 9)
- "Attention and state dynamics are not specified in the paper." (TASK-DOMAINS.md, Summary, line 11)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO decision from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence evidence.
