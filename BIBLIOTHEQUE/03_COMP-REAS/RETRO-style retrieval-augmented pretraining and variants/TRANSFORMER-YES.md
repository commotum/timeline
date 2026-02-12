# Improving language models by retrieving from trillions of tokens (Year not specified)
Source: RETRO-style retrieval-augmented pretraining and variants.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly identifies the core model as a Retrieval-Enhanced Transformer and describes Transformer-style cross-attention as a central mechanism.
- Auxiliary analyses consistently frame the main method as retrieval-enhanced autoregressive language modeling built around Transformer components.

## Evidence
- "With a 2 trillion token database, our Retrieval-Enhanced Transformer (Retro) obtains comparable performance to GPT-3 and Jurassic-1 on the Pile, despite using 25× fewer parameters." (Abstract, RETRO-style retrieval-augmented pretraining and variants.md)
- "Retro combines a frozen Bert retriever, a differentiable encoder and a chunked cross-attention mechanism" (Abstract, RETRO-style retrieval-augmented pretraining and variants.md)
- "We introduce Retro, a retrieval-enhanced autoregressive language model (§2.2)." (Evidence section, TASK-DOMAINS.md)
- "Extending-dimensions analysis markdown: MISSING" (Input availability note; file unavailable in this run)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer classification from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md.
Pass 2 (targeted source scan): skipped - Not needed because Pass 1 was already conclusive.
