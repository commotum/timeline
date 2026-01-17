# Masked-attention Mask Transformer for Universal Image Segmentation (Not specified in the paper.)
Source: Mask2Former- Masked-attention Mask Transformer for Universal Image Segmentation.md

## Core reasons
- Introduces a Transformer-based architecture designed to handle multiple image segmentation tasks (panoptic, instance, semantic), i.e., a 2D vision domain.
- The core contribution is an architectural Transformer decoder with masked attention operating over image features rather than a positional encoding change or a dataset/benchmark.

## Evidence extracts
- "We present Maskedattention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic)." (Abstract)
- "The key components of our Transformer decoder include a masked attention operator, which extracts localized features by constraining crossattention to within the foreground region of the predicted mask for each query, instead of attending to the full feature map." (Section 3.2)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
