# Beyond Brute Force: A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2 (2025)
Source: Beyond Brute Force- A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper explicitly states that its core pipeline includes "transformer-based relational inference," making Transformer mechanisms part of the main model rather than a baseline-only mention.
- The main solving pipeline materially relies on LLM components (o4-mini and Grok-4) for hypothesis generation and final solution generation, indicating a central hybrid architecture with Transformer-family models.

## Evidence
- "We propose a modular architecture combining neural object extraction, transformer-based relational inference, and symbolic rule synthesis in a unified reasoning loop." (Section 1 Introduction, contributions bullet, line 37, Beyond Brute Force- A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2.md)
- "In this work, we demonstrate how our neuro-symbolic reasoning framework augments existing large language models (LLMs) to tackle ARC-AGI-2 tasks far more effectively." (Abstract, line 19, Beyond Brute Force- A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2.md)
- "In the final stage, we leverage Grok-4, not as a pure end-to-end reasoner, but as a constrained generative model guided by our symbolic pipeline." (Section 3.4 Stage 4: LLM Solving with Self-Consistency, line 195, Beyond Brute Force- A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md were read in full; Extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): performed - A targeted architecture scan confirmed explicit transformer usage in the central pipeline.
