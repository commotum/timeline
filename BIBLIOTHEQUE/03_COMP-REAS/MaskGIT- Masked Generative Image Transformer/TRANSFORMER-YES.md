# MaskGIT: Masked Generative Image Transformer (Year not specified)
Source: MaskGIT- Masked Generative Image Transformer.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines MaskGIT as a bidirectional transformer decoder, so Transformer blocks are the core architecture for the main method.
- The abstract describes all-direction token attending during training, which is direct evidence of central self-attention use.

## Evidence
- "This paper proposes a novel image synthesis paradigm using a bidirectional transformer decoder, which we term MaskGIT." (MaskGIT- Masked Generative Image Transformer.md, Abstract, line 11)
- "During training, MaskGIT learns to predict randomly masked tokens by attending to tokens in all directions." (MaskGIT- Masked Generative Image Transformer.md, Abstract, line 11)
- "Based on the bidirectional self-attention and iterative refinement described, the attention is treated as static over the full grid while the state is constructed across iterations." (TASK-DOMAINS.md, Summary, line 14)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer classification; extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 already established that Transformer self-attention is central to the model.
