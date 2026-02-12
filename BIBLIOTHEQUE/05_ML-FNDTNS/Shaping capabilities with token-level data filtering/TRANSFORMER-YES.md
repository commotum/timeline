# Shaping capabilities with token-level data filtering (2026)
Source: Shaping capabilities with token-level data filtering.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The available analysis files explicitly describe the trained core models as Transformers and GPT-2-style architecture, indicating Transformer blocks are central to the main results.
- The abstract frames the work as pretraining-time capability shaping for language models, and the auxiliary model/task analyses tie those experiments directly to Transformer pretraining; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "Current approaches to reducing undesired capabilities in language models are largely post hoc, and can thus be easily bypassed by adversaries. A natural alternative is to shape capabilities during pretraining itself." (Abstract, `Shaping capabilities with token-level data filtering.md`)
- "**Pretraining** We train compute-optimal Transformers at scales ranging from 61M to 1.8B parameters (Hoffmann et al., 2022). Similar to Jordan et al. (2024a), we use an augmented version of the basic GPT-2 architecture (Radford et al., 2019)." (Evidence section, `TASK-DOMAINS.md`, quoting Section 3.2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-YES decision from abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided explicit Transformer/GPT-2 architecture evidence.
