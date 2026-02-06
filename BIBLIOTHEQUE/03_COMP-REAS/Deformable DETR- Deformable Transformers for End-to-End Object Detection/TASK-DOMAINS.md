# DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION (Not specified in the paper.)
Source: Deformable DETR- Deformable Transformers for End-to-End Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Object detection | Images (feature maps) | 2D (x, y) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Bounding boxes and class labels (object detections) | 2D (x, y) (inferred); 0D (inferred) | Fixed (inferred) |

## Summary
The paper presents Deformable DETR as an end-to-end object detection system operating on images. Inputs are 2D spatial feature maps and outputs are sets of bounding boxes with class labels aligned to the image plane. The attention mechanism is data-dependent and the model constructs object-query features as internal state. Input size constraints are not specified; output cardinality is fixed by the number of object queries.

## Evidence
### Task: Object detection
- "Deformable DETR is an end-to-end object detector, which is efficient and fast-converging." (Section 6 Conclusion)
- "Given the input feature maps x ∈ R^{C × H × W} extracted by a CNN backbone" (Section 3 DETR)
- "predict the bounding box coordinates b ∈ [0,1]^4" (Section 3 DETR)
- "The linear projection acts as the classification branch to produce the classification results." (Section 3 DETR)
- "deformable attention module only attends to a small set of key sampling points around a reference point" (Section 4.1)
- "transform the input feature maps to be features of a set of object queries." (Section 3 DETR)
- "input includes both feature maps from the encoder, and N object queries represented by learnable positional embeddings (e.g., N=100)." (Section 3 DETR)
- Inference: In Dimension and Out Dimension mapped to 2D/0D from "x ∈ R^{C × H × W}" and "bounding box coordinates b ∈ [0,1]^4" plus "classification results" (Section 3 DETR). Attention Dynamic inferred from "only attends to a small set of key sampling points" (Section 4.1). State Dynamic inferred from "features of a set of object queries" (Section 3 DETR). Out Dynamics Fixed inferred from "N object queries" (Section 3 DETR).
