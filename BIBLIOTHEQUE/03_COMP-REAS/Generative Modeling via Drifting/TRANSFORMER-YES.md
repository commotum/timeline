# Generative Modeling via Drifting (Year not specified)
Source: Generative Modeling via Drifting.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper explicitly defines the core generator as DiT-like and then specifies a DiT-style Transformer implementation, making Transformer self-attention central to the main model.
- Main reported generation results are produced with this generator architecture, so Transformer blocks are material to primary outcomes rather than peripheral baselines.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but the source file itself provides direct architecture statements sufficient for a confident decision.

## Evidence
- "**Architecture.** Our generator  $(f_{\theta})$  has a DiT-like (Peebles & Xie, 2023) architecture." (Section 4, `Generative Modeling via Drifting.md`, line 232)
- "**Transformer.** We adopt a DiT-style Transformer (Peebles & Xie, 2023)." (Appendix A.2, `Generative Modeling via Drifting.md`, line 533)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Reviewed abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; architecture family was not explicit there, and Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - Found explicit statements that the generator is DiT-like and implemented as a DiT-style Transformer, resolving the decision.
