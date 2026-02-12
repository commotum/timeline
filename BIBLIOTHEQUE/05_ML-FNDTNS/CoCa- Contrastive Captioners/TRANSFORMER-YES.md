# CoCa: Contrastive Captioners are Image-Text Foundation Models (Year not specified)
Source: CoCa- Contrastive Captioners.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly describes a modified encoder-decoder Transformer with decoder cross-attention structure as the core model design.
- Auxiliary analyses are consistent with a Transformer-centered architecture; the Extending-dimensions analysis file was unavailable (`MISSING`) and skipped.

## Evidence
- "In contrast to standard encoder-decoder transformers where all decoder layers attend to encoder outputs, CoCa omits cross-attention in the first half of decoder layers to encode unimodal text representations, and cascades the remaining decoder layers which cross-attend to the image encoder for *multimodal* image-text representations." (Abstract, `CoCa- Contrastive Captioners.md`)
- "Table 1: Size variants of CoCa. Both image encoder and text decoder are Transformers [19, 39]." (Table 1 caption, `CoCa- Contrastive Captioners.md`)
- "captioning loss on the multimodal decoder outputs which predicts text tokens autoregressively." (Evidence section quoting Abstract, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer-based classification.
Pass 2 (targeted source scan): skipped - Not needed after clear abstract/auxiliary Transformer evidence.
