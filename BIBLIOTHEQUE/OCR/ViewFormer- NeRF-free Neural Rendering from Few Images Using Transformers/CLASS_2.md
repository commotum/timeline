# ViewFormer: NeRF-free Neural Rendering from Few Images Using Transformers (Not specified in the paper)
Source: ViewFormer- NeRF-free Neural Rendering from Few Images Using Transformers.md

## Core reasons
- The paper's main contribution is a transformer-based method that maps multiple context images and a pose to synthesize a novel view, adapting transformers to image-based rendering.
- Images are tokenized into a latent sequence and processed by a transformer to generate image tokens, emphasizing a transformer adaptation for visual/scene data rather than positional encoding or datasets.

## Evidence extracts
- "We propose a 2D-only method that maps multiple context views and a query pose to a new image in a single pass of a neural network." (Abstract)
- "For the novel view synthesis task, the transformer is given a set of context views in the code space and the query camera pose, and it generates an image in the code space." (Section 3 Method)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
