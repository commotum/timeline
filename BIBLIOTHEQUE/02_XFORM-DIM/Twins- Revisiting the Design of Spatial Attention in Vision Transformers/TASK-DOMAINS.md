# Twins: Revisiting the Design of Spatial Attention in Vision Transformers (2021)
Source: Twins- Revisiting the Design of Spatial Attention in Vision Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Fixed | Static (inferred) | Direct (inferred) | class labels | 0D (inferred) | Fixed (inferred) |
| Semantic segmentation | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | pixel-wise semantic labels (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Object detection | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | bounding boxes and class labels (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Instance segmentation | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | instance masks and class labels (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates a single-modality vision setting (images) across four task intents: image classification, semantic segmentation, object detection, and instance segmentation. Inputs are consistently image-like 2D spatial domains, while outputs span 0D labels (classification) and 2D spatial predictions (segmentation/detection-related outputs). Input dynamics are explicitly Fixed for ImageNet classification (224×224) and Capped for COCO detection/instance segmentation settings with bounded multi-scale resizing; semantic-segmentation input/output dynamics are not explicitly specified. The attention policy is implemented as predefined global/local self-attention modules (Static, inferred), and the model behavior is treated as reactive feed-forward mapping without explicit persistent external state (Direct, inferred).

## Evidence
### Task: Image classification
- "More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks including image-level classification as well as dense detection and segmentation." (Section Abstract)
- "We first present the ImageNet classification results with our proposed models." (Section 4.1 Classification on ImageNet-1K)
- "All models are trained and evaluated on 224×224 resolution on ImageNet-1K dataset." (Table 1)
- Inference: In Dimension is 2D (x, y) from spatial image formulation ("Given an input of  $H \times W$  resolution" in Section 3.2). Attention Dynamic is Static because attention structure is pre-specified as global or local-global blocks (Section 3). State Dynamic is Direct because the paper describes direct feed-forward backbones/heads without explicit persistent memory/search state. Out Dimension/Out Dynamics are inferred as 0D Fixed from classification intent and Top-1 reporting in Table 1.

### Task: Semantic segmentation
- "We benchmark our proposed architectures on a number of visual tasks, ranging from image-level classification to pixel-level semantic/instance segmentation and object detection." (Section 1 Introduction)
- "We test on the ADE20K dataset [42], a challenging scene parsing task for semantic segmentation" (Section 4.2 Semantic Segmentation on ADE20K)
- "This dataset contains 20K images for training and 2K images for validation." (Section 4.2 Semantic Segmentation on ADE20K)
- Inference: Input/Output dimensions are 2D (x, y) from image-based scene parsing/semantic segmentation wording (Section 4.2) and pixel-level segmentation wording (Section 1). Output is inferred as pixel-wise semantic labels from the task name and mIoU evaluation. Attention Dynamic and State Dynamic are inferred as Static and Direct for the same architectural reasons as above. In/Out Dynamics are not explicitly specified for this task configuration.

### Task: Object detection
- "#### 4.3 Object Detection and Segmentation on COCO" (Section 4.3 Object Detection and Segmentation on COCO)
- "We evaluate the performance of our method using two representative frameworks: RetinaNet [46] and Mask RCNN [47]. Specifically, we use our transformer models to build the backbones of these detectors." (Section 4.3 Object Detection and Segmentation on COCO)
- "For  $1\times$  schedule object detection with RetinaNet, Twins-PCPVT-S surpasses PVT-Small with 2.6% mAP" (Section 4.3 Object Detection and Segmentation on COCO)
- Inference: In Dynamics is Capped from explicit bounded resize policy ("randomly resizing the input image so that its shorter side is between 480 and 800 while keeping longer one less than 1333" in Section 4.3). In/Out Dimension are inferred as 2D (x, y) for image-plane detection outputs; output is inferred as bounding boxes and class labels from detection framework/task context. Attention Dynamic is inferred Static and State Dynamic inferred Direct from fixed-module transformer backbone design.

### Task: Instance segmentation
- "We benchmark our proposed architectures on a number of visual tasks, ranging from image-level classification to pixel-level semantic/instance segmentation and object detection." (Section 1 Introduction)
- "For  $1\times$  object segmentation with the Mask R-CNN framework, Twins-PCPVT-S brings similar improvements (+2.5% mAP) over PVT-Small." (Section 4.3 Object Detection and Segmentation on COCO)
- "**Table 4** – Object detection and instance segmentation performance on the COCO val2017 dataset using the Mask R-CNN framework." (Table 4)
- Inference: In Dynamics is Capped by the same bounded COCO multi-scale resize policy in Section 4.3. In/Out Dimension are inferred as 2D (x, y) because instance masks are spatial image-plane outputs; output is inferred as instance masks and class labels from Mask R-CNN instance-segmentation context. Attention Dynamic is inferred Static and State Dynamic inferred Direct from the predefined attention blocks and absence of explicit persistent external state.
