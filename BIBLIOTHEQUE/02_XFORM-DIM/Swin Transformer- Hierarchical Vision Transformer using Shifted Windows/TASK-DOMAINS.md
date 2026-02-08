# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Year not specified in the paper.)
Source: Swin Transformer- Hierarchical Vision Transformer using Shifted Windows.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | RGB images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class label | 0D (inferred) | Fixed (inferred) |
| object detection | RGB images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | bounding boxes | 2D (x, y) (inferred) | Capped (inferred) |
| instance segmentation | RGB images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | instance masks | 2D (x, y) (inferred) | Capped (inferred) |
| semantic segmentation | RGB images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | semantic segmentation map (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates four vision tasks: image classification, object detection, instance segmentation, and semantic segmentation. All tasks use image inputs, which map to 2D (x, y) domains, while outputs span 0D labels (classification) and 2D spatial predictions (boxes/masks/segmentation maps). Dynamics are supported as Fixed for ImageNet classification and Capped for COCO detection/instance segmentation, while semantic-segmentation dynamics are not explicitly specified in the OCR text. Across tasks, attention is supported as Static (fixed local/shifted windows) and state as Constructed (hierarchical feature representations), both marked as inferences from the architecture description.

## Evidence
### Task: image classification
- "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4)
- "For image classification, we benchmark the proposed Swin Transformer on ImageNet-1K [18], which contains 1.28M training images and 50K validation images from 1,000 classes." (Section 4.1)
- "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)
- Inference: `2D (x, y)` input and `0D` output are inferred from RGB-image input plus classification intent; `Fixed` dynamics are inferred from fixed single-crop evaluation and listed fixed image sizes ("The top-1 accuracy on a single crop is reported." (Section 4.1), Table 1 image sizes). `Static` attention and `Constructed` state are inferred from fixed windowed attention ("we propose to compute self-attention within local windows." (Section 3.2)) and hierarchical representation building ("Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers." (Section 1)).

### Task: object detection
- "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4)
- "Object detection and instance segmentation experiments are conducted on COCO 2017, which contains" (Section 4.2)
- "118K training, 5K validation and 20K test-dev images." (Section 4.2)
- "multi-scale training [8, 52] (resizing the input such that the shorter side is between 480 and 800 while the longer side is at most 1333)" (Section 4.2)
- Inference: `2D (x, y)` input/output are inferred from image-domain detection and box metrics (`AP^box` tables); `Capped` input dynamics are inferred from explicit resize bounds in Section 4.2, and output `Capped` is inferred as finite per-image detections under box-AP evaluation. `Static` attention and `Constructed` state are inferred from "we propose to compute self-attention within local windows." (Section 3.2) and "Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers." (Section 1).

### Task: instance segmentation
- "These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val)." (Abstract)
- "Object detection and instance segmentation experiments are conducted on COCO 2017, which contains" (Section 4.2)
- "Table 2. Results on COCO object detection and instance segmentation." (Table 2 caption)
- Inference: `2D (x, y)` input and mask output dimensions are inferred from image-domain instance segmentation and `AP^mask` reporting; `Capped` dynamics are inferred from COCO bounded resize settings and finite per-image instance-mask predictions. `Static` attention and `Constructed` state are inferred from "we propose to compute self-attention within local windows." (Section 3.2) and "Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers." (Section 1).

### Task: semantic segmentation
- "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4)
- "ADE20K [74] is a widely-used semantic segmentation dataset, covering a broad range of 150 semantic categories." (Section 4.3)
- "There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level" (Section 1)
- Inference: `2D (x, y)` input/output and semantic segmentation map output are inferred from RGB-image input plus pixel-level dense prediction phrasing in Section 1. `In Dynamics` and `Out Dynamics` are marked "Not specified in the paper." because explicit segmentation input/output size constraints are deferred ("More details are presented in the Appendix." (Section 4.3)) and not provided in this OCR markdown. `Static` attention and `Constructed` state are inferred from "we propose to compute self-attention within local windows." (Section 3.2) and "Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers." (Section 1).
