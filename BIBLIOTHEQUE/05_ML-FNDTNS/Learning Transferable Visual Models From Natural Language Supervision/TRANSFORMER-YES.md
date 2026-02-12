# Learning Transferable Visual Models From Natural Language Supervision (Year not specified)
Source: Learning Transferable Visual Models From Natural Language Supervision.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper explicitly states that one of the core CLIP image encoder options is Vision Transformer (ViT), which is a Transformer architecture used in main experiments.
- The paper explicitly defines CLIP's text encoder as a Transformer and states masked self-attention is used, making Transformer-style self-attention central to the model family.

## Evidence
- "For the second architecture, we experiment with the recently introduced Vision Transformer (ViT) (Dosovitskiy et al., 2020)." (Learning Transferable Visual Models From Natural Language Supervision.md, Section 2.4, line 106)
- "The text encoder is a Transformer (Vaswani et al., 2017) ... Masked self-attention was used in the text encoder ..." (Learning Transferable Visual Models From Natural Language Supervision.md, Section 2.4, line 108)
- "The attention pooling is implemented as a single layer of \"transformer-style\" multi-head QKV attention ..." (Learning Transferable Visual Models From Natural Language Supervision.md, Section 2.4, line 77)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - abstract and auxiliary files were reviewed fully; they established task breadth but did not provide enough explicit architecture evidence for a high-confidence Transformer-family decision. The extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - targeted model-section scan found explicit ViT image encoder usage plus Transformer text encoder with masked self-attention.
