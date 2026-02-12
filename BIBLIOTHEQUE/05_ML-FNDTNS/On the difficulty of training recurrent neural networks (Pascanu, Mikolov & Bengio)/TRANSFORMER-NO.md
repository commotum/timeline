# On the difficulty of training recurrent neural networks (2013)
Source: On the difficulty of training recurrent neural networks (Pascanu, Mikolov & Bengio).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and task analyses center on recurrent neural networks, gradient clipping, and vanishing/exploding gradients, not Transformer-style self-attention.
- The extending-dimensions analysis file was unavailable (`MISSING`), and the available auxiliary files still provide sufficient evidence for a confident non-Transformer classification.

## Evidence
- "There are two widely known issues with properly training recurrent neural networks, the vanishing and the exploding gradient problems detailed in Bengio et al. (1994)." (Abstract in `On the difficulty of training recurrent neural networks (Pascanu, Mikolov & Bengio).md`)
- "Attention dynamics are not described" (Summary in `TASK-DOMAINS.md`)
- "Not specified in the paper." (attention_dynamic column entries in `TASK-DOMAINS.csv`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence classification.
