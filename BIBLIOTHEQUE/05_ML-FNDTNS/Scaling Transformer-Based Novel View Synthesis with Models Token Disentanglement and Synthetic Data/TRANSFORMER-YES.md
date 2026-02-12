# Scaling Transformer-Based Novel View Synthesis with Models Token Disentanglement and Synthetic Data (Year not specified)
Source: Scaling Transformer-Based Novel View Synthesis with Models Token Disentanglement and Synthetic Data.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states that the proposed method is transformer-based and introduces a transformer-architecture modification (Tok-D) as the main contribution.
- Auxiliary analyses are consistent with a single feed-forward NVS model built around transformer blocks rather than a non-attention backbone.

## Evidence
- "Large transformer-based models have made significant progress in generalizable novel view synthesis (NVS) from sparse input views, generating novel viewpoints without the need for test-time optimization." (Abstract, `Scaling Transformer-Based Novel View Synthesis with Models Token Disentanglement and Synthetic Data.md`)
- "To address this, we propose a token disentanglement process within the transformer architecture, enhancing feature separation and ensuring more effective learning." (Abstract, `Scaling Transformer-Based Novel View Synthesis with Models Token Disentanglement and Synthetic Data.md`)
- "Finally, the transformer network is trained to reconstruct the target output tokens  $O_i^t$  from the Plücker patch embeddings." (Section 3 quote captured in `TASK-DOMAINS.md` Evidence)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for TRANSFORMER-YES from abstract and auxiliary files; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence evidence.
