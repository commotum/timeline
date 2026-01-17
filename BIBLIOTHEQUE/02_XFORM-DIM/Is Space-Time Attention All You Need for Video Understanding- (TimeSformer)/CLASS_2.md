# Is Space-Time Attention All You Need for Video Understanding? (Not specified in the paper.)
Source: Is Space-Time Attention All You Need for Video Understanding- (TimeSformer).md

## Core reasons
- Adapts a Transformer (ViT) from images to videos by modeling spatiotemporal patches with space-time self-attention.
- Central contribution is enabling Transformer attention over a higher-dimensional video domain, not proposing a new positional encoding mechanism.

## Evidence extracts
- "We present a convolution-free approach to video classification built exclusively on self-attention over space and time. Our method, named "TimeSformer," adapts the standard Transformer architecture to video by enabling spatiotemporal feature learning directly from a sequence of framelevel patches." (Abstract)
- "We adapt the image model "Vision Transformer" (ViT) (Dosovitskiy et al., 2020) to video by extending the self-attention mechanism from the image space to the space-time 3D volume." (Section 1. Introduction)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
