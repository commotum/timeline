# How Can Self-Attention Networks Recognize Dyck-n Languages? (Year not specified)
Source: How Can Self-Attention Networks Recognize Dyck-n Languages-.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly says the study focuses on Dyck-n recognition "with self-attention (SA) networks" and reports results for SA variants.
- The auxiliary analysis identifies the trained core architecture as "two multi-headed self-attention networks (i.e., only the encoder part of a Transformer)," making Transformer-style self-attention central to the main results.

## Evidence
- "We focus on the recognition of Dyck-n  $(\mathcal{D}_n)$ languages with self-attention (SA) networks" (Abstract, How Can Self-Attention Networks Recognize Dyck-n Languages-.md)
- "We train two multi-headed self-attention networks (i.e., only the encoder part of a Transformer), one of which incorporates an additional starting symbol in the vocabulary (SA+), and the other does not (SA-)." (Section 3 Experiments quote captured in TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions analysis file was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient for a high-confidence decision.
