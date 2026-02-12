# Scaling and evaluating sparse autoencoders (2024)
Source: Scaling and evaluating sparse autoencoders.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s core method is a sparse autoencoder that reconstructs language-model activations via encoder/decoder bottlenecks, not a self-attention architecture.
- Transformer models (e.g., GPT-2/GPT-4) are used as upstream activation sources to analyze, while the trained/evaluated model in this paper is the autoencoder itself.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but abstract plus available auxiliary files already provide sufficient architecture evidence.

## Evidence
- "Sparse autoencoders provide a promising unsupervised approach for extracting interpretable features from a language model by reconstructing activations from a sparse bottleneck layer." (Abstract, `Scaling and evaluating sparse autoencoders.md`)
- "For an input vector  $x \in \mathbb{R}^d$  from the residual stream ... $$\hat{x} = W_{\text{dec}}z + b_{\text{pre}}$$" (Task evidence quoting Section 2.2, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence NO decision from abstract and auxiliary analyses.
Pass 2 (targeted source scan): skipped - Not needed after Pass 1; decision already clear.
