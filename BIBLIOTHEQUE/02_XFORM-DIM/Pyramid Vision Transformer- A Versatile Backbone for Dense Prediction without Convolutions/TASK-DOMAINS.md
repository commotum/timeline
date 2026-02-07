# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions (Not specified in the paper)
Source: Pyramid Vision Transformer- A Versatile Backbone for Dense Prediction without Convolutions.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels | 0D (inferred) | Fixed (inferred) |
| object detection | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | bounding boxes and class labels | 2D (x, y) (inferred); 0D (inferred) | Not specified in the paper. |
| instance segmentation | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | instance masks | 2D (x, y) (inferred) | Not specified in the paper. |
| semantic segmentation | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | semantic segmentation map | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper evaluates PVT on four image tasks: classification, object detection, instance segmentation, and semantic segmentation. All tasks use 2D image inputs with fixed (classification) or capped (detection/segmentation) sizing based on the reported cropping/resizing regimes (inferred). Attention is applied over the given image-derived features (static) and the system is a feed-forward mapping without persistent state (direct) (inferred).

## Evidence
### Task: classification
- "Image classification is the most classical task of imagelevel prediction." (Section 4.1. Image-Level Prediction)
- "append a learnable classification token to the input of the last stage, and then employ a fully connected (FC) layer to conduct classification" (Section 4.1. Image-Level Prediction)
- Inference: Input/dimension and fixed sizing inferred from "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture) and "a  $224 \times$ 224 patch is cropped to evaluate the classification accuracy." (Section 5.1. Image Classification); attention/state labeled static/direct from "our SRA receives a query Q, a key K, and a value V as input, and outputs a refined feature." (Section 3.3. Transformer Encoder)

### Task: object detection
- "Here, we discuss two typical tasks, namely object detection, and semantic segmentation." (Section 4.2. Pixel-Level Dense Prediction)
- "Object detection experiments are conducted on the challenging COCO benchmark [40]." (Section 5.2. Object Detection)
- Inference: Image input/dimension from "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture); capped sizing from "The training image is resized to have a shorter side of 800 pixels, while the longer side does not exceed 1,333 pixels." (Section 5.2. Object Detection); attention/state labeled static/direct from "our SRA receives a query Q, a key K, and a value V as input, and outputs a refined feature." (Section 3.3. Transformer Encoder)

### Task: instance segmentation
- "Similar results are found in instance segmentation experiments based on Mask R-CNN, as shown in Table 4." (Section 5.2. Object Detection)
- "Table 4: **Object detection and instance segmentation performance on COCO val2017.**" (Table 4 caption)
- Inference: Image input/dimension from "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture); capped sizing from "The training image is resized to have a shorter side of 800 pixels, while the longer side does not exceed 1,333 pixels." (Section 5.2. Object Detection); attention/state labeled static/direct from "our SRA receives a query Q, a key K, and a value V as input, and outputs a refined feature." (Section 3.3. Transformer Encoder)

### Task: semantic segmentation
- "Here, we discuss two typical tasks, namely object detection, and semantic segmentation." (Section 4.2. Pixel-Level Dense Prediction)
- "We choose ADE20K [83], a challenging scene parsing dataset, to benchmark the performance of semantic segmentation." (Section 5.3. Semantic Segmentation)
- Inference: Image input/dimension from "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture); capped sizing and output map dynamics inferred from "We randomly resize and crop the image to  $512 \times 512$  for training, and rescale to have a shorter side of 512 pixels during testing." (Section 5.3. Semantic Segmentation) and "pixel-level classification or regression to be performed on the feature map" (Section 4.2. Pixel-Level Dense Prediction); attention/state labeled static/direct from "our SRA receives a query Q, a key K, and a value V as input, and outputs a refined feature." (Section 3.3. Transformer Encoder)
