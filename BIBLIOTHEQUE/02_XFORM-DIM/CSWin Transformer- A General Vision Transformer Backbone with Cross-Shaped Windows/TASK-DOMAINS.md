# CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows (Not specified in the paper.)
Source: CSWin Transformer- A General Vision Transformer Backbone with Cross-Shaped Windows.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Not specified in the paper. |
| Object detection | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Not specified in the paper. | bounding boxes with class labels (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Instance segmentation | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Not specified in the paper. | instance masks (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Semantic segmentation | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Not specified in the paper. | semantic segmentation map (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates a single vision modality (images) across image classification, object detection, instance segmentation, and semantic segmentation tasks. Inputs are 2D images and outputs are either 0D class labels or 2D spatial predictions, inferred from the task descriptions and reported metrics. Attention is inferred to be static due to the fixed cross-shaped window self-attention design, while dynamics and state are largely not specified in the paper.

## Evidence
### Task: Image classification
- "we conduct experiments on ImageNet-1K [16] classification, COCO [37] object detection, and ADE20K [72] semantic segmentation." (Section 4. Experiments)
- "#### 4.1. ImageNet-1K Classification" (Section 4.1)
- Inference: Input is a 2D image and output is a single class label; attention is static from fixed stripe windows. Supported by "For an input image with size of  $H \times W \times 3$ ," (Section 3.1), "Top-1 accuracy on ImageNet-1K" (Abstract), and "Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel" (Abstract).

### Task: Object detection
- "we conduct experiments on ImageNet-1K [16] classification, COCO [37] object detection, and ADE20K [72] semantic segmentation." (Section 4. Experiments)
- "Next, we evaluate CSWin Transformer on the COCO objection detection task with the Mask R-CNN [21] and Cascade Mask R-CNN [2] framework respectively." (Section 4.2)
- Inference: Input is a 2D image and outputs are bounding boxes with class labels; attention is static from fixed stripe windows. Supported by "For an input image with size of  $H \times W \times 3$ ," (Section 3.1), "box AP" and "COCO detection task" (Abstract), and "Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel" (Abstract).

### Task: Instance segmentation
- "Object detection and instance segmentation performance on the COCO val2017 with the Mask R-CNN framework." (Table 4 caption)
- "mask AP" (Abstract)
- Inference: Input is a 2D image and outputs are instance masks; attention is static from fixed stripe windows. Supported by "For an input image with size of  $H \times W \times 3$ ," (Section 3.1), "mask AP" (Abstract), and "Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel" (Abstract).

### Task: Semantic segmentation
- "ADE20K semantic segmentation task" (Abstract)
- "We further investigate the capability of CSWin Transformer for Semantic Segmentation on the ADE20K [72] dataset." (Section 4.3)
- Inference: Input is a 2D image and output is a 2D semantic label map; attention is static from fixed stripe windows. Supported by "For an input image with size of  $H \times W \times 3$ ," (Section 3.1), "we report the results of different methods in terms of mIoU" (Section 4.3), and "Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel" (Abstract).
