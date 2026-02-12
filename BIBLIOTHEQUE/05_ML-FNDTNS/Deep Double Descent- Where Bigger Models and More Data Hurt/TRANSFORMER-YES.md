# Deep Double Descent: Where Bigger Models and More Data Hurt (Year not specified)
Source: Deep Double Descent- Where Bigger Models and More Data Hurt.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s main experiments explicitly include Transformer models as one of the core architecture families, not just related-work mentions.
- Available auxiliary analyses tie key reported results to multi-head-attention encoder-decoder Transformer experiments.
- The extending-dimensions analysis file was unavailable (`MISSING`), so the decision is based on the abstract plus the other available auxiliary files.

## Evidence
- "We show that a variety of modern deep learning tasks exhibit a \"double-descent\" phenomenon where, as we increase model size, performance first gets *worse* and then gets better." (Abstract, `Deep Double Descent- Where Bigger Models and More Data Hurt.md`)
- "Transformers on language translation tasks: Multi-head-attention encoder-decoder Transformer model" (Evidence section, `TASK-DOMAINS.md`, quoting Figure 8 caption)
- "We consider three families of architectures: ResNets, standard CNNs, and Transformers." (`TASK_MODEL_RATIO.md`, item 2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient for a high-confidence decision.
