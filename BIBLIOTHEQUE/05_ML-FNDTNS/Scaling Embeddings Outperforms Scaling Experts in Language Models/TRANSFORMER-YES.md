# Scaling Embeddings Outperforms Scaling Experts in Language Models (Year not specified)
Source: Scaling Embeddings Outperforms Scaling Experts in Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the work as a large-language-model MoE architecture, and the auxiliary analysis shows attention modules in the model internals.
- `TASK-DOMAINS.md` contains an explicit quote about the model’s first attention module output, indicating attention is central to the architecture used for results.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract + auxiliary evidence was sufficient.

## Evidence
- "Mixture-of-Experts (MoE) architectures have become the standard for sparsity scaling in large language models" (Abstract, `Scaling Embeddings Outperforms Scaling Experts in Language Models.md`)
- "the L2 norm of the first attention module's output is an order of magnitude larger" (Section 3.2.4 quote recorded in `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence Transformer classification; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already sufficient.
