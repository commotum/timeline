# Rethinking and Improving Relative Position Encoding for Vision Transformer (Year not specified in the paper.)
Source: Rethinking and Improving Relative Position Encoding for Vision Transformer (iRPE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Object detection | Images | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Bounding boxes | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper evaluates two vision tasks: image classification and object detection, both using image inputs. The supported input modality is consistently 2D spatial data (2D (x, y)). Based on the described model interfaces, classification is fixed-size while detection is capped by explicit image-size and query limits (inferred). The attention/state profiles are best supported as Static attention and Direct state for both tasks (inferred from full-input self-attention pipelines without runtime retrieval/memory construction).

## Evidence
### Task: Image classification
- "Then, we compare the proposed methods with the state-of-the-art methods on image classification and object detection tasks." (Section 4. Experiments)
- "We compare our proposed methods with the state-of-the-art methods on image classification tasks." (Section 4.3. Comparison on Image Classification)
- "For training, the images are split into 14x14 non-overlapping patches." (Section 4.1. Implementation Details)
- "In ViT [6] and DeiT [22] models, an image is split into multiple fixed-size patches. ... An extra trainable classification token is added into the sequence for classification." (Section 5. Related Work)
- Inference: `In Dynamics = Fixed` is inferred from fixed-size patching and fixed classification input settings; `Attention Dynamic = Static` is inferred from full-sequence self-attention over patch tokens rather than runtime selection; `State Dynamic = Direct` is inferred from one-pass mapping without explicit persistent memory; `Output = Class label`, `Out Dimension = 0D`, and `Out Dynamics = Fixed` are inferred from the stated classification setup and top-1/top-5 classification evaluation. (Supported by Section 4.1, Section 4.3, Section 5)

### Task: Object detection
- "To verify the generality of our method, we further evaluate it on COCO 2017 detection dataset [12]." (Section 4.4. Comparison on Object Detection)
- "We use the transformer-based detection model DETR [1] as our baseline." (Section 4.4. Comparison on Object Detection)
- "The transformer outputs a certain number of bounding boxes." (Section 5. Related Work)
- "The image is cropped such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333. ... The number of queries is 100." (Section 5. Training and Test Settings of DETR)
- Inference: `In Dynamics = Capped` is inferred from explicit bounded image sizing for DETR preprocessing; `Attention Dynamic = Static` is inferred from standard transformer self-attention over provided tokens/queries without runtime retrieval policy; `State Dynamic = Direct` is inferred from the direct image-to-detection forward pipeline; `Out Dimension = 2D (x, y)` and `Out Dynamics = Capped` are inferred from bounding-box localization with a fixed query budget. (Supported by Section 4.4, Section 5 Related Work, Section 5 Training and Test Settings of DETR)
