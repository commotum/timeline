# Taming Transformers for High-Resolution Image Synthesis (Not specified in the paper.)
Source: Taming Transformers for High-Resolution Image Synthesis.md

## Core reasons
- The paper adapts transformers to high-resolution image synthesis by combining CNN-based codebooks with transformer modeling to handle 2D image data efficiently.
- It explicitly converts images into sequences of discrete codebook indices so a transformer can model image compositions, which is a dimensional adaptation beyond 1D text.

## Evidence extracts
- "We demonstrate how combining the effectiveness of the inductive bias of CNNs with the expressivity of transformers enables them to model and thereby synthesize high-resolution images." (Abstract)
- "To utilize the highly expressive transformer architecture for image synthesis, we need to express the constituents of an image in the form of a sequence." (Section 3.1)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
