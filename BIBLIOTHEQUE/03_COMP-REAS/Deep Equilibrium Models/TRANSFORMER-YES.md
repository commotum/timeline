# Deep Equilibrium Models (Year not specified)
Source: Deep Equilibrium Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly states that a central DEQ instantiation uses self-attention transformers, not just as a baseline or citation.
- Main reported results include the DEQ-Transformer path, so Transformer-style self-attention is materially part of the paper’s core modeling contribution.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient to decide.

## Evidence
- "We demonstrate how DEQs can be applied to two state-of-the-art deep sequence models: self-attention transformers and trellis networks." (Abstract, Deep Equilibrium Models.md)
- "On large-scale language modeling tasks, such as the WikiText-103 benchmark, we show that DEQs 1) often improve performance over these stateof-the-art models" (Abstract, Deep Equilibrium Models.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-YES from abstract plus TASK-DOMAINS/TASK-DOMAINS.csv/TASK_MODEL_RATIO; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture evidence.
