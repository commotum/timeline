# Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge (Year not specified)
Source: Logic Tensor Networks- Deep Learning and Logical Reasoning from Data and Knowledge.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes the core implementation as "deep Tensor Neural Networks" in TensorFlow and does not indicate Transformer/self-attention blocks as a central mechanism.
- Auxiliary analyses consistently report no attention mechanism signal ("Not specified in the paper"); the extending-dimensions analysis file was unavailable (`MISSING`) and was therefore skipped.

## Evidence
- "We show how Real Logic can be implemented in deep Tensor Neural Networks with the use of Google's TensorFlow<sup>™</sup> primitives." (Abstract, `Logic Tensor Networks- Deep Learning and Logical Reasoning from Data and Knowledge.md`)
- "while attention dynamics are not specified." (Summary, `TASK-DOMAINS.md`)
- "knowledge completion,knowledge-base facts and logical constraints over objects/relations,0D (inferred),Open (inferred),Not specified in the paper.,Constructed (inferred),truth-values for known and missing facts; groundings (feature vectors) for constants,0D; 1D (t) (inferred),Open (inferred)" (`TASK-DOMAINS.csv`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence NO decision.
Pass 2 (targeted source scan): skipped - abstract + auxiliary files were sufficient; extending-dimensions file was unavailable (`MISSING`).
