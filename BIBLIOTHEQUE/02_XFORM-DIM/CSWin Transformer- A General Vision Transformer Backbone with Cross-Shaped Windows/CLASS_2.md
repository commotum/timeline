# CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows (Not specified in the paper.)
Source: CSWin Transformer- A General Vision Transformer Backbone with Cross-Shaped Windows.md

## Core reasons
- Proposes a Transformer backbone explicitly for vision tasks, targeting efficient attention for image tokens rather than only sequence domains.
- Describes processing 2D images into patch tokens and applying cross-shaped window attention across horizontal/vertical stripes, indicating a Transformer adaptation for 2D data.

## Evidence extracts
- "We present CSWin Transformer, an efficient and effective Transformer-based backbone for general-purpose vision tasks." (Abstract)
- "For an input image with size of  $H \times W \times 3$ , we follow [60] and leverage the overlapped convolutional token embedding ( $7 \times 7$  convolution layer with stride 4)) to obtain  $\frac{H}{4} \times \frac{W}{4}$  patch tokens" (Section 3.1. Overall Architecture)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
