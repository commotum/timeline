# Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation (Year not specified in the paper.)
Source: Stand-Alone Axial-Attention for Panoptic Segmentation (Axial-DeepLab).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | class label (Top-1) (inferred) | 0D (inferred) | Fixed (inferred) |
| Panoptic segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | panoptic segmentation | 2D (x, y) (inferred) | Not specified in the paper. |
| Instance segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | class-agnostic instance segmentation | 2D (x, y) (inferred) | Not specified in the paper. |
| Semantic segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | semantic segmentation | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper explicitly evaluates four tasks: image classification, panoptic segmentation, instance segmentation, and semantic segmentation, all on image inputs. The supported input structure is 2D (x, y), and outputs span 0D class labels for classification and 2D segmentation outputs for dense prediction tasks. Attention is classified as Static (inferred) because the model attends over predefined regions/axes (local span or whole feature extent), and State is Direct (inferred) because the paper describes feed-forward mappings without persistent constructed task state. Input/output Dynamics are mostly not explicitly specified in the OCR text, except a fixed single-label classification output.

## Evidence
### Task: Image classification
- "We show the effectiveness of our axial-attention models on ImageNet [70] for classification, and on three datasets (COCO [56], Mapillary Vistas [62], and Cityscapes [22]) for panoptic segmentation [45], instance segmentation, and semantic segmentation." (Section 1 Introduction)
- "We first report results with our Axial-ResNet on ImageNet [70]." (Section 4 Experimental Results)
- "Table 1. ImageNet validation set results." and "Top-1" (Section 4.1 ImageNet)
- Inference: `Attention Dynamic = Static` is inferred from "For each location o, a local  m \times m  square region is extracted" and "setting the span m directly to the whole input features" (Section 3.1 and Section 3.2), indicating predefined attention scope at design time. `State Dynamic = Direct` is inferred because the method is presented as stacked feed-forward attention blocks without persistent memory/state objects. `Output = class label (Top-1)`, `Out Dimension = 0D`, and `Out Dynamics = Fixed` are inferred from the classification task framing and Top-1 evaluation.

### Task: Panoptic segmentation
- "We then convert the ImageNet pretrained Axial-ResNet to Axial-DeepLab, and report results on COCO [56], Mapillary Vistas [62], and Cityscapes [22] for panoptic segmentation, evaluated by panoptic quality (PQ) [45]." (Section 4 Experimental Results)
- "The heads produce semantic segmentation and class-agnostic instance segmentation, and they are merged by majority voting [89] to form the final panoptic segmentation." (Section 3.2 Axial-DeepLab)
- Inference: `Attention Dynamic = Static` and `State Dynamic = Direct` use the same architectural evidence as above (Section 3.1 and Section 3.2). `Out Dimension = 2D (x, y)` is inferred because panoptic output is formed by merging per-pixel semantic and instance segmentation predictions.

### Task: Instance segmentation
- "We show the effectiveness of our axial-attention models on ImageNet [70] for classification, and on three datasets (COCO [56], Mapillary Vistas [62], and Cityscapes [22]) for panoptic segmentation [45], instance segmentation, and semantic segmentation." (Section 1 Introduction)
- "We also report average precision (AP) for instance segmentation, and mean IoU for semantic segmentation on Mapillary Vistas and Cityscapes." (Section 4 Experimental Results)
- "The heads produce semantic segmentation and class-agnostic instance segmentation" (Section 3.2 Axial-DeepLab)
- Inference: `Attention Dynamic = Static` and `State Dynamic = Direct` are inferred from fixed attention span/axes and feed-forward design (Section 3.1 and Section 3.2). `Out Dimension = 2D (x, y)` is inferred because instance segmentation is produced as image-space segmentation outputs.

### Task: Semantic segmentation
- "We show the effectiveness of our axial-attention models on ImageNet [70] for classification, and on three datasets (COCO [56], Mapillary Vistas [62], and Cityscapes [22]) for panoptic segmentation [45], instance segmentation, and semantic segmentation." (Section 1 Introduction)
- "We also report average precision (AP) for instance segmentation, and mean IoU for semantic segmentation on Mapillary Vistas and Cityscapes." (Section 4 Experimental Results)
- "The heads produce semantic segmentation and class-agnostic instance segmentation" (Section 3.2 Axial-DeepLab)
- Inference: `Attention Dynamic = Static` and `State Dynamic = Direct` are inferred from the predefined attention scope and feed-forward architecture (Section 3.1 and Section 3.2). `Out Dimension = 2D (x, y)` is inferred because semantic segmentation output is a dense image-space prediction.
