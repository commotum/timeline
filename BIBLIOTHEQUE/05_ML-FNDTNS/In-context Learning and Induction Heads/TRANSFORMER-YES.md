# In-context Learning and Induction Heads (2022)
Source: In-context Learning and Induction Heads.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly centers "induction heads" (attention heads) in "large transformer models" and frames the main claim as mechanistic in-context learning in transformers.
- Auxiliary analyses also describe transformer models as the core setting; the extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract + auxiliary evidence is sufficient for a high-confidence decision.

## Evidence
- "\"Induction heads\" are attention heads that implement a simple algorithm to complete token sequences like  $[A][B] ... [A] \rightarrow [B]$ . In this work, we present preliminary and indirect evidence for a hypothesis that induction heads might constitute the mechanism for the majority of all \"incontext learning\" in large transformer models" (In-context Learning and Induction Heads.md:17, Abstract)
- "In this argument, we show some anecdotal examples of induction heads from larger transformers (our 40-layer model with 13 billion parameters)" (TASK_MODEL_RATIO.md:3, quoting Argument 4)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-YES from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture evidence.
