# Video Swin Transformer (Not specified in the paper.)
Source: Video Swin Transformer.md

## Core reasons
- The paper adapts an image-domain Transformer (Swin Transformer) to video by extending attention from spatial to spatiotemporal windows, making the central contribution a higher-dimensional Transformer architecture for video recognition.
- It tokenizes video into 3D patches and applies 3D windowed/shifted self-attention, showing the core adaptation to spatiotemporal inputs rather than a positional encoding innovation.

## Evidence extracts
- "Our model, called Video Swin Transformer, strictly follows the hierarchical structure of the original Swin Transformer, but extends the scope of local attention computation from only the spatial domain to the spatiotemporal domain." (Section 1 Introduction)
- "In Video Swin Transformer, we treat each 3D patch of size  $2 \times 4 \times 4 \times 3$  as a token." (Section 3.1 Overall Architecture)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
