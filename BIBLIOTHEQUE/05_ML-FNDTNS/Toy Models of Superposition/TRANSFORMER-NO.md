# Toy Models of Superposition (2022)
Source: Toy Models of Superposition.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes the core method as toy ReLU networks on sparse synthetic features, not Transformer blocks or self-attention.
- The auxiliary task/domain analysis explicitly characterizes attention dynamics as static and the architecture as feed-forward ReLU.
- The Extending-dimensions analysis file was unavailable (`MISSING`), but the abstract and available auxiliary files already provide sufficient architecture evidence.

## Evidence
- "In this paper, we use toy models — small ReLU networks trained on synthetic data with sparse input features — to investigate how and when models represent more features than they have dimensions." (Toy Models of Superposition.md, Abstract)
- "The architectures are feed-forward ReLU models with no runtime mechanism for selecting different input slices, so Attention is Static and State is Direct (both inferred)." (TASK-DOMAINS.md, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO decision from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Not needed after high-confidence Pass 1 evidence.
