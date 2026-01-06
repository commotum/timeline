# End-to-End Object Detection with Transformers (DETR) (Not specified in the paper.)
Source: End-to-End Object Detection with Transformers (DETR).md

## Core reasons
- The paper adapts a Transformer encoder-decoder architecture to object detection, applying attention-based sequence modeling to images rather than 1D text sequences.
- It explicitly converts 2D image features into a sequence with positional encoding so the Transformer can operate over image spatial structure.

## Evidence extracts
- "The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture." (Abstract)
- "DETR uses a conventional CNN backbone to learn a 2D representation of an input image. The model flattens it and supplements it with a positional encoding before passing it into a transformer encoder." (Section 3.2, Fig. 2)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
