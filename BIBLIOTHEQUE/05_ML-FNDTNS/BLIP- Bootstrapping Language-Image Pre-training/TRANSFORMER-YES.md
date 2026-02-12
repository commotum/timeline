# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Year not specified)
Source: BLIP- Bootstrapping Language-Image Pre-training.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The model architecture explicitly uses a visual transformer (ViT) as the image encoder.
- The core multimodal text components are transformer blocks with self-attention and cross-attention, and the decoder uses causal self-attention.

## Evidence
- "We employ a visual transformer (Dosovitskiy et al., 2021) as our image encoder... using a ViT is more computation-friendly" (BLIP- Bootstrapping Language-Image Pre-training.md, Section 3.1 Model Architecture)
- "Image-grounded text encoder... inserting one additional cross-attention (CA) layer between the self-attention (SA) layer and the feed forward network (FFN) for each transformer block of the text encoder." (BLIP- Bootstrapping Language-Image Pre-training.md, Section 3.1 Model Architecture)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Abstract and auxiliary files were read in full; transformer evidence was not explicit enough in auxiliaries, and the Extending-dimensions file was unavailable (MISSING).
Pass 2 (targeted source scan): performed - Targeted scan of architecture lines in the source confirmed ViT + self-attention/cross-attention transformer blocks as core model components.
