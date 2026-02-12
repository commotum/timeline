# Hierarchical Reasoning Model (Year not specified)
Source: Hierarchical Reasoning Model (HRM).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly defines HRM as a recurrent architecture with two recurrent modules, not a Transformer/self-attention backbone.
- Auxiliary analysis files consistently characterize state as recurrent/constructed and attention as static/inferred, with no indication that Transformer-style self-attention is a central model component.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency." (Abstract, `Hierarchical Reasoning Model (HRM).md:9`)
- "through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations." (Abstract, `Hierarchical Reasoning Model (HRM).md:9`)
- "The model uses recurrent hidden states and fixed token sequences, so Attention is treated as Static and State as Constructed where supported by the architectural description (inferred)." (`TASK-DOMAINS.md:12`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-NO from the abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture evidence.
