# DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning (Year not specified)
Source: DreamCoder.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes DreamCoder as Bayesian program induction with a learned symbolic language and a neural recognition model; it does not describe Transformer blocks or self-attention as core architecture.
- Auxiliary analyses indicate attention signals are unspecified rather than central, which is inconsistent with a Transformer-centered method.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract + auxiliary files are sufficient for a high-confidence decision.

## Evidence
- "It builds expertise by creating programming languages for expressing domain concepts, together with neural networks to guide the search for programs within these languages." (DreamCoder.md:7, abstract)
- "A \"wake-sleep\" learning algorithm alternately extends the language with new symbolic abstractions and trains the neural network on imagined and replayed problems." (DreamCoder.md:7, abstract)
- "The paper explicitly fixes the number of examples in some domains (list processing and regex), but leaves other dynamics and attention/state characteristics unspecified." (TASK-DOMAINS.md:17, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided clear architectural evidence and no Transformer/self-attention core.
