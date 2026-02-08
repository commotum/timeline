# Transformer in Transformer (Not specified in the paper.)
Source: Transformer in Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Class labels (inferred) | 0D (inferred) | Fixed (inferred) |
| Object detection | Images | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Object bounding boxes and class labels (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Semantic segmentation | Images | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Pixel-wise semantic labels (inferred) | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper evaluates TNT on three visual task intents: image classification, object detection, and semantic segmentation. All tasks operate on image inputs, which the paper explicitly frames as 2D and patch-tokenized, while outputs span a 0D class decision space (classification) and 2D spatial predictions (detection and segmentation, inferred). Input dynamics are fixed for classification settings and bounded/capped for dense tasks through explicit resizing/cropping constraints. Across tasks, TNT uses static attention over predefined tokenized inputs and constructed internal state via sentence/word embedding memories (both inferred from the architecture description).

## Evidence
### Task: Image classification
- "ImageNet ILSVRC 2012 [30] is an image classification benchmark consisting of 1.2M training images belonging to 1000 classes, and 50K validation images with 50 images per class." (Section 3.1 Datasets and Experimental Settings)
- "Given a 2D image, we uniformly split it into n patches  $\mathcal{X} = [X^1, X^2, \cdots, X^n] \in \mathbb{R}^{n \times p \times p \times 3}$ , where (p,p) is the resolution of each image patch." (Section 2.2 Transformer in Transformer)
- "Finally, the classification token serves as the image representation and a fully-connected layer is applied for classification." (Section 2.2 Transformer in Transformer)
- Inference: `In Dynamics = Fixed` is inferred from fixed-resolution settings ("processing a 224 × 224 image" in Section 2.4 and "All models are fine-tuned with an image resolution of 384×384" in Section 3.5). `Attention Dynamic = Static` is inferred because TNT computes self-attention over predefined patch/word token sequences from the input image (Section 2.2). `State Dynamic = Constructed` is inferred from "we create the sentence embedding memories to store the sequence of sentence-level representations" (Section 2.2). `Output = Class labels`, `Out Dimension = 0D`, and `Out Dynamics = Fixed` are inferred from the explicit classification setup and class-token classification head (Sections 2.2, 3.1, 3.5).

### Task: Object detection
- "Pure Transformer Object Detection. We construct a pure transformer object detection pipeline by combining our TNT and DETR [3]." (Section 3.5 Transfer Learning)
- "The training images are randomly resized to have a shorter side in the range of [640,800] and a longer side within 1333 pixels." (Section 3.5 Transfer Learning)
- "Table 10: Results of object detection on COCO2017 val set with ImageNet pre-training." (Section 3.5 Transfer Learning)
- Inference: `In Dynamics = Capped` is inferred from explicit bounded resizing rules in training/testing (Section 3.5). `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from the same TNT token-attention and embedding-memory design described in Section 2.2. `Output = Object bounding boxes and class labels` and `Out Dimension = 2D (x, y)` are inferred from the object-detection task framing and COCO AP reporting. `Out Dynamics` is not explicitly specified in the paper.

### Task: Semantic segmentation
- "Pure Transformer Semantic Segmentation. We adopt the segmentation framework of Trans2Seg [42] to build the pure transformer semantic segmentation based on TNT backbone." (Section 3.5 Transfer Learning)
- "We apply random resize and crop of 512×512 during training." (Section 3.5 Transfer Learning)
- "Table 11: Results of semantic segmentation on ADE20K val set with ImageNet pre-training." (Section 3.5 Transfer Learning)
- Inference: `In Dynamics = Capped` is inferred from resize/crop-constrained preprocessing in the segmentation setup (Section 3.5). `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from the TNT architecture in Section 2.2. `Output = Pixel-wise semantic labels`, `Out Dimension = 2D (x, y)`, and `Out Dynamics = Capped` are inferred from the semantic-segmentation task framing and mIoU evaluation on image grids (Section 3.5).
