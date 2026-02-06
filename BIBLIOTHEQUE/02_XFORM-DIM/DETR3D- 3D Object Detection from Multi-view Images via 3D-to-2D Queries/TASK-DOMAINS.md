# DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries (Not specified in the paper)
Source: DETR3D- 3D Object Detection from Multi-view Images via 3D-to-2D Queries.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-view 3D object detection | Multi-view RGB images + camera projection matrices | 2D (x, y) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | 3D bounding boxes + class labels | 3D (x, y, z); 0D | Fixed (inferred) |

## Summary
The paper focuses on a single task: multi-view 3D object detection from RGB camera images, producing 3D bounding boxes with class labels. The input modality is 2D images (with camera projection matrices), while the outputs are 3D spatial boxes plus 0D labels. The evidence supports fixed input/output dynamics in the evaluated setup, and the attention/state classifications are inferred from the model's query-based feature sampling and iterative object-query refinement.

## Evidence
### Task: Multi-view 3D object detection
- "We introduce a framework for multi-camera 3D object detection." (Abstract)
- "Our architecture inputs RGB images collected from a set of cameras whose projection matrices (the combination of intrinsics and relative extrinsics) are known" (Section 3.1 Overview)
- "it outputs a set of 3D bounding box parameters for the objects in the scene." (Section 3.1 Overview)
- "our model aims to predict these boxes and their labels from the these images." (Section 3.2 Feature Learning)
- Inference: In Dynamics is Fixed and Out Dynamics is Fixed because the evaluated setup uses a fixed six-camera input and a fixed number of prediction slots ("Each sample contains images from 6 cameras [front_left, front, front_right, back_left, back, back_right]." (Section 4.1 Implementation Details); "The number of ground-truth boxes M is typically smaller than the number of predictions  $M^*$ , so we pad the set of ground-truth boxes with  $\varnothing$ s (no object) up to  $M^*$  for ease of computation." (Section 3.4 Loss)). Attention Dynamic is Dynamic because the model selects image features based on projected reference points at runtime ("project these centers into all the feature maps using the camera transformation matrices;" and "sample features via bilinear interpolation and incorporate them into object queries;" (Section 3.3 Detection Head)). State Dynamic is Constructed because the model maintains learned object priors and iteratively refines object queries ("Our method starts from a sparse set of object priors, shared across the dataset and learned end-to-end." (Section 1 Introduction); "This layer is repeated multiple times, alternating between feature sampling and object query refinement." (Section 3.1 Overview)).
