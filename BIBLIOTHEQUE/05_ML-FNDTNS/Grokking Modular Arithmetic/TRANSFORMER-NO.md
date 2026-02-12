# Grokking modular arithmetic (Year not specified)
Source: Grokking Modular Arithmetic.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the main method as "fully-connected two-layer networks," indicating an MLP-centric approach rather than Transformer self-attention.
- Auxiliary task/domain files characterize the model as a "two-layer MLP network without biases" with static/non-attention dynamics; Transformer mentions appear as prior-work context, not the central architecture.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available abstract and auxiliary evidence is sufficient for a high-confidence decision.

## Evidence
- "We present a simple neural network that can learn modular arithmetic tasks and exhibits a sudden jump in generalization known as \"grokking\". Concretely, we present (i) fully-connected two-layer networks that exhibit grokking on various modular arithmetic tasks..." (`Grokking Modular Arithmetic.md`, Abstract)
- "We consider a two-layer MLP network without biases" (`TASK-DOMAINS.md`, Evidence section)
- "Based on the fixed-size one-hot interface and two-layer MLP, the tasks use 1D fixed vectors with static attention and direct state (inferred)." (`TASK-DOMAINS.md`, Summary)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
