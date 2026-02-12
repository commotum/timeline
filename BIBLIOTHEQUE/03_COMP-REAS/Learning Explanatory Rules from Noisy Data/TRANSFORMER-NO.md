# Learning Explanatory Rules from Noisy Data (Year not specified)
Source: Learning Explanatory Rules from Noisy Data.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract presents the central method as Differentiable Inductive Logic Programming and describes hybridization with neural networks, not Transformer/self-attention blocks.
- Auxiliary analyses characterize the work as ILP/predicate-learning focused and do not provide Transformer or attention-centric model cues; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "In this paper, we propose a Differentiable Inductive Logic framework" (Abstract, `Learning Explanatory Rules from Noisy Data.md`:7)
- "We tested  $\partial$ ILP on 20 ILP tasks" (Section 5.3 quote captured in `TASK_MODEL_RATIO.md`:3)
- "Not specified in the paper." (Attention Dynamic entries in `TASK-DOMAINS.md`:7 and `TASK-DOMAINS.csv`:2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence non-Transformer classification from the abstract and auxiliary analyses; extending-dimensions input was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient for final decision.
