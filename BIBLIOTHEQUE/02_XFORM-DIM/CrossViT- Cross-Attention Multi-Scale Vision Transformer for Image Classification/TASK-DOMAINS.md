# CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification (Not specified in the paper.)
Source: CrossViT- Cross-Attention Multi-Scale Vision Transformer for Image Classification.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
CrossViT is presented as a vision transformer for image classification and is evaluated on ImageNet1K with transfer learning on five additional image-classification datasets spanning natural and medical images. The paper describes images tokenized into patches and fixed evaluation crops (e.g., 224 x 224), supporting a 2D (x, y) input dimension with Fixed dynamics (inferred). Outputs are single class predictions (0D, Fixed; inferred), and the model applies attention over all tokens without runtime selection, indicating Static attention and Direct state (inferred).

## Evidence
### Task: image classification
- "in this paper, we study how to learn multi-scale feature representations in transformer models for image classification." (Abstract)
- "We validate this by performing transfer learning on 5 image classification tasks" (Transfer Learning)
- "While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)
- "Vision Transformer (ViT) [11] first converts an image into a sequence of patch tokens by dividing it with a certain patch size" (Section 3.1 Overview of Vision Transformer)
- "Afterwards, all tokens are passed through stacked transformer encoders and finally the CLS token is used for classification." (Section 3.1 Overview of Vision Transformer)
- "take the center crop  $224 \times 224$  as the input." (Section 4.1 Experimental Setup)
- Inference: Input dimension is 2D (x, y) and input dynamics are Fixed because the task operates on images and uses fixed crops ("Vision Transformer (ViT) [11] first converts an image into a sequence of patch tokens"; "take the center crop  $224 \times 224$  as the input."). Output is treated as a single class label (0D, Fixed) and attention/state are Static/Direct because "all tokens are passed through stacked transformer encoders" and "the CLS token is used for classification."
