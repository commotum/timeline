# AN IMAGE IS WORTH 16x16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (Not specified in the paper.)
Source: An Image is Worth 16×16 Words- Transformers for Image Recognition at Scale (ViT).md

## Core reasons
- The paper's main contribution is adapting a standard Transformer to images by treating image patches as a token sequence for classification.
- It positions the work as applying Transformers beyond 1D language into 2D vision, which is central to the method rather than a positional encoding innovation.

## Evidence extracts
- "We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks." (Abstract)
- "To handle 2D images, we reshape the image  $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$  into a sequence of flattened 2D patches  $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$" (Section 3.1 VISION TRANSFORMER (VIT))

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
