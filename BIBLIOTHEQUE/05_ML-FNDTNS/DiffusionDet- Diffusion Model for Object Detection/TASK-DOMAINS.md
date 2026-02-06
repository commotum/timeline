# DiffusionDet: Diffusion Model for Object Detection (Not specified in the paper)
Source: DiffusionDet- Diffusion Model for Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| object detection | images | 2D (x, y) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | bounding boxes and category labels | 2D (x, y); 0D (inferred) | Capped (inferred) |

## Summary
DiffusionDet is presented as an object detection system that takes images and predicts bounding boxes with category labels. The paper describes a diffusion-based, iterative refinement process from noisy boxes to object boxes, implying dynamic attention over RoI regions and constructed intermediate state during inference. The number of output boxes can vary at evaluation time, while the paper does not specify fixed input size constraints; dimensions are 2D spatial with 0D labels (inferred).

## Evidence
### Task: object detection
- "Object detection aims to predict a set of bounding boxes and associated category labels for targeted objects in one image." (Section 1. Introduction)
- "x is the input image, b and c are a set of bounding boxes and category labels for objects in the image x, respectively." (Section 3.1. Preliminaries)
- "we can train DiffusionDet with  $N_{train}$  random boxes while evaluating it with  $N_{eval}$  random boxes, where the  $N_{eval}$  is arbitrary" (Section 1. Introduction)
- Inference: In/Out Dimension marked 2D (x, y) because boxes are defined as " $b^i = (c_x^i, c_y^i, w^i, h^i)$ " (Section 3.1. Preliminaries) and outputs are spatially indexed; 0D is inferred for category labels. Attention Dynamic marked Dynamic because the decoder "takes as input a set of proposal boxes to crop RoI-feature" and the model "progressively refines its predictions" (Sections 3.2, 3.4). State Dynamic marked Constructed because inference is a "denoising sampling process from noise to object boxes" (Section 3.4). Out Dynamics marked Capped because evaluation uses an arbitrary but chosen N_eval ("the  $N_{eval}$  is arbitrary") (Section 1. Introduction).

---
## CSV Output (required)
```csv
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
object detection,images,"2D (x, y) (inferred)",Not specified in the paper.,Dynamic (inferred),Constructed (inferred),"bounding boxes and category labels","2D (x, y); 0D (inferred)",Capped (inferred)
```
