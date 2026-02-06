# ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices (Not specified in the paper.)
Source: ComRoPE- Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (2D images) | images | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | class labels (inferred) | 0D (inferred) | Fixed (inferred) |
| object detection | images (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | object detections (bounding boxes + labels) (inferred) | 2D (x, y) (inferred) | Capped (inferred) |
| classification (3D video) | video frames (inferred) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates ComRoPE on vision tasks: 2D image classification on ImageNet-1K, object detection on MS COCO, and 3D classification on UCF-101. Input dimensionality spans 2D (x, y) images and 3D (x, y, t) video clips (inferred), with outputs framed as classification labels or detection results (inferred). Input dynamics for 3D classification are inferred from the fixed frame count, while other dynamics and the attention/state characterizations are inferred from the use of standard Vision Transformer self-attention.

## Evidence
### Task: classification (2D images)
- "We first assess their scalability in 2D image classification across different resolutions." (Section 4. Experiments)
- "Table 1. Accuracy of 2D classification on ImageNet." (Section 4. Experiments)
- "on the ImageNet-1K dataset [5]." (Section 4.1.1 Setup)
- Inference: Input dynamics marked Capped because models are "trained at a standard resolution of  $224 \times 224$  and evaluated across multiple resolutions"; attention/state marked Static/Direct because they "utilize a standard Vision Transformer (ViT-B/16)" with "self-attention layers"; outputs marked as class labels (0D, Fixed) because this is "2D image classification." (Sections 4, 4.1.1)

### Task: object detection
- "Additionally, we conduct object detection experiments to demonstrate the generalizability of our approach." (Section 4. Experiments)
- "We evaluate ComRoPE-LD, LieRE, and APE on the MS COCO dataset [17]." (Section 4.2. Object detection)
- "Table 2. Results of object detection on MS COCO." (Section 4.2. Object detection)
- Inference: Inputs treated as 2D images and outputs as 2D detections (with capped multiplicity) based on the "object detection" framing on MS COCO and the vision backbone statement "We adopt ViT-S as our backbone and apply Com-RoPE to the attention layers"; attention/state marked Static/Direct from use of attention layers. (Section 4.2)

### Task: classification (3D video)
- "we perform 3D classification experiments, which are detailed in Appendix B." (Section 4. Experiments)
- "we conduct a 3D classification task on UCF-101 [31]." (Appendix B.1)
- "Table 3. Accuracy of 3D classification on UCF-101." (Appendix B)
- Inference: Inputs treated as 3D video (x, y, t) with Fixed dynamics based on "Frame Count      | 8" and "Image Size       | 224"; outputs marked as class labels (0D, Fixed) because this is a "3D classification" task; attention/state marked Static/Direct based on transformer attention head constraints in "the head dimension be a multiple coordinate dimension" discussion. (Appendix C.2)
