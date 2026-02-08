# Rotary Position Embedding for Vision Transformer (Year not specified in the paper)
Source: Rotary Position Embedding for Vision Transformer (RoPE‑Mixed).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-resolution classification | images | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | class label | 0D | Fixed |
| Object detection | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | bounding boxes and class labels | 2D (x, y); 0D | Not specified in the paper. |
| Semantic segmentation | images | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | segmentation map (per-pixel labels) | 2D (x, y) | Fixed (inferred) |

## Summary
The paper covers three vision tasks on images: multi-resolution classification, object detection, and semantic segmentation. The input domain is consistently 2D (x, y), while outputs span 0D class labels, 2D+0D detections, and 2D segmentation maps. Dynamics vary by task setting in the paper: classification is evaluated across bounded resolution ranges, segmentation is reported with 512 × 512 inputs, and detection input/output bounds are not explicitly specified. Attention and state are not directly labeled in glossary terms, but the described ViT/Swin processing supports Static attention and Direct state as inferences.

## Evidence
### Task: Multi-resolution classification
- "RoPE in ViT and Swin Transformer is validated for image recognition, including multi-resolution classification (§4.1) on ImageNet-1k [4], object detection (§4.2) on MS-COCO [16], and semantic segmentation (§4.3) on ADE20k [40,41]." (Section 4 Experiments)
- "We report the accuracy on the ImageNet-1k validation set as varying image sizes. Note that we use the ImageNet-1k standard image resolution  $224 \times 224$  for training. Thus, a resolution larger than 224 can be considered as extrapolation." (Section 4.1 Multi-resolution classification)
- Inference: Input is 2D (x, y) from the paper’s explicit 2D image framing ("RoPE for 2D images" in Section 3.2). In Dynamics is marked Capped (inferred) because classification is explicitly evaluated over a bounded set of image sizes (Section 4.1; Appendix Table A.3/A.4/A.5). Attention Dynamic is Static (inferred) and State Dynamic is Direct (inferred) because the model consumes a predefined tokenized image through self-attention rather than runtime retrieval/constructed external state (Sections 3 and 4.1). Output is a class label (0D) inferred from the classification task framing and accuracy reporting (Section 4.1).

### Task: Object detection
- "RoPE in ViT and Swin Transformer is validated for image recognition, including multi-resolution classification (§4.1) on ImageNet-1k [4], object detection (§4.2) on MS-COCO [16], and semantic segmentation (§4.3) on ADE20k [40,41]." (Section 4 Experiments)
- "We verify 2D RoPE in object detection on MS-COCO [16]. DINO [39] detector is trained using ViT and Swin as backbone network." (Section 4.2 Object detection)
- "Table 1 shows the DINO-ViTDet results in bounding box AP." (Section 4.2 Object detection)
- Inference: Input is 2D (x, y) from the 2D image domain description (Section 3.2). Output is represented as bounding boxes and class labels with Out Dimension 2D (x, y); 0D (inferred from object detection intent and "bounding box AP" in Section 4.2). Attention Dynamic is Static (inferred) based on fixed attention structures ("window-block attention" and remaining global layers in Section 4.2). State Dynamic is Direct (inferred) because the paper describes feed-forward detection with pretrained backbones rather than explicit constructed memory/state.

### Task: Semantic segmentation
- "RoPE in ViT and Swin Transformer is validated for image recognition, including multi-resolution classification (§4.1) on ImageNet-1k [4], object detection (§4.2) on MS-COCO [16], and semantic segmentation (§4.3) on ADE20k [40,41]." (Section 4 Experiments)
- "We train 2D RoPE ViT and Swin for semantic segmentation on ADE20k [40, 41]." (Section 4.3 Semantic segmentation)
- "The improvement might originate from the extrapolation performance of RoPE since the ViT-UperNet setting uses 512 × 512 images for inputs." (Section 4.3 Semantic segmentation)
- Inference: Output is a segmentation map (per-pixel labels) with Out Dimension 2D (x, y), inferred from the semantic segmentation task definition and mIoU reporting in Tables 3 and 4 (Section 4.3). Out Dynamics is Fixed (inferred) from the stated fixed 512 × 512 input setting in this experiment (Section 4.3). Attention Dynamic is Static (inferred) and State Dynamic is Direct (inferred) for the same architectural reason as above (ViT/Swin self-attention pipeline without explicit runtime retrieval/state construction).
