# The Capacity for Moral Self-Correction in Large Language Models (2023)
Source: The Capacity for Moral Self-Correction in Large Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s core study is on large language models trained with RLHF, and all reported main results are from that model family.
- The available auxiliary analyses explicitly cite the methods statement that the evaluated models are decoder-only transformer models.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the remaining Pass 1 evidence is explicit and sufficient.

## Evidence
- "We test the hypothesis that language models trained with reinforcement learning from human feedback (RLHF) have the capability to \"morally self-correct\"..." (Abstract, The Capacity for Moral Self-Correction in Large Language Models.md)
- "We study decoder-only transformer models fine-tuned with Reinforcement Learning from Human Feedback (RLHF) [13, 57] to function as helpful dialogue models." (Section 3.1 quote captured in TASK-DOMAINS.md and TASK_MODEL_RATIO.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence decision from the abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided explicit architecture evidence.
