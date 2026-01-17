# Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (Not specified in the paper.)
Source: SETR- Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers.md

## Core reasons
- Recasts semantic segmentation as sequence-to-sequence and applies a Transformer encoder to image patch sequences, adapting Transformers to a 2D vision task.
- Introduces image-to-sequence patch tokenization to feed 2D images into a Transformer for dense prediction, which is a dimensional lifting contribution rather than a positional encoding change.

## Evidence extracts
- "In this paper, we aim to provide an alternative perspective by treating semantic segmentation as a sequence-to-sequence prediction task." (Abstract)
- "we divide an image  $x \in \mathbb{R}^{H \times W \times 3}$  into a grid of  $\frac{H}{16} \times \frac{W}{16}$  patches uniformly, and then flatten this grid into a sequence." (Section 3.2)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
