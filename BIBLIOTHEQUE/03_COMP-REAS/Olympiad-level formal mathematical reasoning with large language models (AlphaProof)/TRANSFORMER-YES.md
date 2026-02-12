# Olympiad-level formal mathematical reasoning with reinforcement learning (2025)
Source: Olympiad-level formal mathematical reasoning with large language models (AlphaProof).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- `TASK_MODEL_RATIO.md` directly identifies the core proof network as an encoder-decoder Transformer, making self-attention central to the main system.
- The abstract describes AlphaProof as the core reasoning engine for the reported IMO-level results, so the Transformer-based proof network is materially central to the paper's primary outcomes.
- Extending-dimensions analysis markdown was unavailable (`MISSING`), but the available abstract and auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "At its core is the proof network, a 3 billion parameters encoder-decoder transformer model [15, 16], that learns to interpret the observed Lean tactic state (fig. 1b) and generate two outputs: a policy, suggesting promising tactics to apply next, and a value function, estimating the expected return  $G_t$  (as defined in section 1.1)." (`TASK_MODEL_RATIO.md`, quote attributed to Section 1.2 "Prover Agent")
- "At the 2024 IMO competition, our AI system, with AlphaProof as its core reasoning engine, solved three out of the five non-geometry problems, including the competition's most difficult problem." (`Olympiad-level formal mathematical reasoning with large language models (AlphaProof).md`, Abstract)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence TRANSFORMER-YES decision.
Pass 2 (targeted source scan): skipped - not needed because Pass 1 already established the architecture signal.
