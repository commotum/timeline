# Exploring Plain Vision Transformer Backbones for Object Detection (Year not specified in the paper)
Source: ViTDet- Exploring Plain ViT Backbones for Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bounding-box object detection | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Bounding boxes and class labels | 2D (x, y) (inferred) | Capped (inferred) |
| Instance segmentation | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Instance masks and class labels | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper evaluates two image tasks: bounding-box object detection and instance segmentation. The reported inputs are images with fixed resolutions in the training/evaluation recipes ("1024 × 1024" and, for one COCO system-level setting, "from 1024 to 1280"), supporting 2D input with Fixed dynamics. Outputs are spatial instance predictions in the image plane, and the detector output count is explicitly limited ("≤ 300 detections per image" and "COCO's default 100"), supporting Capped output dynamics. The backbone/head pipeline uses predefined window/full-image processing and constructed intermediate representations (feature pyramid, RPN, RoI heads), supporting Static attention and Constructed state.

## Evidence
### Task: Bounding-box object detection
- "We report results on bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)." (Section 4.1 Ablation Study and Analysis)
- "The input image is  $1024 \times 1024$ , augmented with large-scale jittering [19] during training." (Implementation)
- "We output  $\leq 300$  detections per image following [23] (vs. COCO's default 100)." (Appendix A.2, Hyper-parameters for LVIS)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from image-plane detection and box metrics in Section 4.1 (AP<sup>box</sup>). `In Dynamics = Fixed` is inferred from fixed input-size statements (Implementation; Section 4.3 notes "increase the input size (from 1024 to 1280)"). `Out Dynamics = Capped` is inferred from the explicit detection-count cap in Appendix A.2. `Attention Dynamic = Static` is inferred from predefined processing ("we divide it into regular non-overlapping windows. Self-attention is computed within each window.", Section 3). `State Dynamic = Constructed` is inferred from explicit intermediate structures ("simple feature pyramid", "RPN", and "RoI heads", Sections 3 and 4.1).

### Task: Instance segmentation
- "We report results on bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)." (Section 4.1 Ablation Study and Analysis)
- "LVIS contains ~2M high-quality instance segmentation annotations for 1203 classes that exhibit a natural, long-tailed object distribution." (Section 4.3 Comparisons on LVIS)
- "We output  $\leq 300$  detections per image following [23] (vs. COCO's default 100)." (Appendix A.2, Hyper-parameters for LVIS)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred because instance masks are image-plane outputs (Sections 4.1 and 4.3). `In Dynamics = Fixed` is inferred from fixed-resolution recipes (Implementation and Section 4.3). `Out Dynamics = Capped` is inferred from the explicit per-image detection cap in Appendix A.2. `Attention Dynamic = Static` is inferred from fixed window/global attention placement during fine-tuning (Section 3). `State Dynamic = Constructed` is inferred from the model’s explicit constructed intermediate representations (simple feature pyramid with RPN/RoI processing; Sections 3 and 4.1).
