# DeepProbLog: Neural Probabilistic Logic Programming (Year not specified)
Source: DeepProbLog- Neural Probabilistic Logic Programming.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes DeepProbLog as probabilistic logic programming extended with neural predicates, not a self-attention/Transformer architecture.
- Across the auxiliary task/model files, attention is marked as not specified and the explicit neural architecture cue is CNN; the extending-dimensions analysis file was unavailable (MISSING).

## Evidence
- "We introduce DeepProbLog, a probabilistic logic programming language that incorporates deep learning by means of neural predicates." (DeepProbLog- Neural Probabilistic Logic Programming.md, Abstract, line 24)
- "Not specified in the paper." (TASK-DOMAINS.csv, `attention_dynamic` field across task rows, lines 2-7)
- "The CNN does not generalize to this variable-length problem setting." (TASK-DOMAINS.md, Task: addition (multi-digit MNIST sum), line 26)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence NO decision from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions file unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
