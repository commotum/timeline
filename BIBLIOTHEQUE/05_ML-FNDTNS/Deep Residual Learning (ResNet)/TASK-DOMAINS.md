# Deep Residual Learning for Image Recognition (2015)
Source: Deep Residual Learning (ResNet).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Fixed (inferred) |
| Object detection | images | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Not specified in the paper. | bounding boxes and class labels (inferred) | 2D (x, y) (inferred); 0D (inferred) | Capped (inferred) |
| Image localization (classification + localization) | images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | per-class classification and box regression outputs | 2D (x, y) (inferred); 0D (inferred) | Fixed (inferred) |
| Segmentation (COCO) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper covers image classification on ImageNet/CIFAR-10 and extends the same residual networks to object detection and ImageNet localization, while also reporting competition results for COCO segmentation. For classification, detection, and localization, the inputs are images (2D) and outputs include class predictions and, for detection/localization, bounding boxes; segmentation I/O details are not specified. Dynamics are described as fixed or capped where explicit (e.g., fixed-size crops or a fixed set of image scales and proposal counts), while attention and state dynamics are largely not specified.

## Evidence
### Task: Image classification
- "We evaluate our method on the ImageNet 2012 classification dataset [36] that consists of 1000 classes." (Section 4.1 ImageNet Classification)
- "The network ends with a global average pooling layer and a 1000-way fully-connected layer with softmax." (Section 3.3 Network Architectures)
- "A  $224 \times 224$  crop is randomly sampled from an image or its horizontal flip" (Section 3.4 Implementation)
- Inference: Input images and fixed-size crops imply 2D (x, y) with Fixed input dynamics; a fixed number of classes implies 0D fixed outputs.

### Task: Object detection
- "We adopt *Faster R-CNN* [32] as the detection method." (Section 4.3 Object Detection on PASCAL and MS COCO)
- "These layers are shared by a region proposal network (RPN, generating 300 proposals) [32] and a Fast R-CNN detection network [7]." (Appendix A. Object Detection Baselines)
- "The final classification layer is replaced by two sibling layers (classification and box regression [7])." (Appendix A. Object Detection Baselines)
- "the image's shorter sides are  $s \in \{200, 400, 600, 800, 1000\}$ ." (Section B. Object Detection Improvements)
- Inference: Images imply 2D (x, y); the fixed set of scales implies Capped input dynamics; RPN-generated proposals imply Dynamic attention and Capped output dynamics; classification + box regression implies bounding boxes and class labels with 2D/0D outputs.

### Task: Image localization (classification + localization)
- "The ImageNet Localization (LOC) task [36] requires to classify and localize the objects." (Section C. ImageNet Localization)
- "we adopt the \"per-class regression\" (PCR) strategy [40, 41], learning a bounding box regressor for each class." (Section C. ImageNet Localization)
- "the *cls* layer has a 1000-d output" (Section C. ImageNet Localization)
- "the *reg* layer has a  $1000\times4$ -d output consisting of box regressors for 1000 classes." (Section C. ImageNet Localization)
- Inference: Images imply 2D (x, y) input; fixed-size 1000-d and 1000x4-d outputs imply Fixed output dynamics and 0D/2D output dimensions.

### Task: Segmentation (COCO)
- "we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation." (Abstract)
