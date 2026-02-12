# DEPTH-ADAPTIVE TRANSFORMER (Year not specified)
Source: Depth-Adaptive Transformers.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly states the paper trains Transformer models as the main method, not just as a baseline.
- The task/domain analysis explicitly identifies standard Transformer self-/cross-attention in the model description.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract and available auxiliary files were already sufficient for a high-confidence decision.

## Evidence
- "In this paper, we train Transformer models which can make output predictions at different stages of the network and we investigate different ways to predict how much computation is required for a particular sequence." (Depth-Adaptive Transformers.md, Abstract)
- "the model uses standard Transformer self-/cross-attention" (TASK-DOMAINS.md, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision.
Pass 2 (targeted source scan): skipped - not needed because Pass 1 was already conclusive.
