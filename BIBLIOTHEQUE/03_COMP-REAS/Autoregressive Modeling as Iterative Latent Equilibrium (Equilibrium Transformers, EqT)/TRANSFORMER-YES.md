# Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT) (2025)
Source: Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines EqT as a modification of standard Transformer layers, indicating Transformer architecture is central to the paper’s main method.
- Auxiliary analysis explicitly references multi-head self-attention within the EqT block, confirming material use of Transformer-style self-attention in the core model.
- Extending-dimensions analysis markdown was unavailable (`MISSING`), but abstract plus available auxiliary files already provide direct architecture evidence.

## Evidence
- "We instantiate this principle as Equilibrium Transformers (EqT), which augment standard transformer layers with an Equilibrium Refinement Module" (Abstract, `Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT).md`)
- "The standard transformer block consists of multi-head self-attention (MHSA) followed by a feed-forward network (FFN):" (Section 2.3 excerpt, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-YES decision from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md.
Pass 2 (targeted source scan): skipped - Not needed because Pass 1 was already conclusive.
