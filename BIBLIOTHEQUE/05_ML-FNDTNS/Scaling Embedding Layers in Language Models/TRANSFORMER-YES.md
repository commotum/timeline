# Scaling Embedding Layers in Language Models (2025)
Source: Scaling Embedding Layers in Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper’s core method (SCONE) is built around Transformer architectures: it augments a standard Transformer main model and uses a separate f-gram Transformer model during training.
- Auxiliary analysis confirms decoder-only language modeling as the primary task family, and the `Extending-dimensions` file was unavailable (`MISSING`), so the decision relies on available abstract/auxiliary evidence plus targeted architecture lines from the source.

## Evidence
- "We propose to augment a standard transformer model with an additional f-gram embedding layer." (Section 3 SCONE Architecture, `Scaling Embedding Layers in Language Models.md`)
- "During training, it is parameterized by an f-gram transformer model" (Section 3 SCONE Architecture, `Scaling Embedding Layers in Language Models.md`)
- "We focus on pre-training decoder-only language models with causal language modeling [Radford et al., 2019]." (Evidence section, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - suggestive Transformer-family cues found, but abstract alone was not explicit enough on self-attention architecture.
Pass 2 (targeted source scan): performed - direct method statements confirm Transformer models are central.
