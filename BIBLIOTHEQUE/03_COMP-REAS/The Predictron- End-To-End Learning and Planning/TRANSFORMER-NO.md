# The Predictron: End-To-End Learning and Planning (2017)
Source: The Predictron- End-To-End Learning and Planning.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines a predictron built around an internal Markov reward process and planning rollouts, not Transformer-style self-attention blocks.
- Available auxiliary analysis describes constructed internal state and adaptive depth via lambda-style gating, and does not indicate self-attention as a core mechanism.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "The predictron consists of a fully abstract model, represented by a Markov reward process, that can be rolled forward multiple \"imagined\" planning steps." (The Predictron- End-To-End Learning and Planning.md, Abstract)
- "All variants utilise a convolutional core with 2 intermediate hidden layers; parameters were updated by supervised learning (see appendix for more details)." (The Predictron- End-To-End Learning and Planning.md, Section 5.2)
- "The architecture constructs internal abstract state and uses state-dependent gating for adaptive computation depth, supporting Constructed state and Dynamic attention (inferred)." (TASK-DOMAINS.md, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-NO decision.
Pass 2 (targeted source scan): skipped - Pass 1 already decisive.
