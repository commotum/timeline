# Point-voxel dual transformer for LiDAR 3D object detection (2025)
Source: Point-voxel dual transformer for LiDAR 3D object detection (PV-DT3D).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiDAR 3D object detection | LiDAR point cloud (3D points with coordinates and reflectance) | 3D (x, y, z) | Fixed | Static (inferred) | Constructed (inferred) | 3D bounding boxes and confidence scores | 3D (x, y, z); 0D | Capped |

## Summary
The paper presents a single-task LiDAR 3D object detection system evaluated on KITTI, operating on 3D point cloud scenes and producing 3D bounding box detections with confidence scores. The input interface is fixed via point sampling, and the output is capped by selecting a top set of proposals at inference. Attention is treated as static (inferred) and state as constructed (inferred) because the model processes a fixed keypoint set and builds a global proposal representation for prediction.

## Evidence
### Task: LiDAR 3D object detection
- "a two-stage light detection and ranging (LiDAR) three-dimensional (3D) object detection framework" (Opening paragraph before Section 1. Introduction)
- "given an *N*-points 3D scene with position coordinates and reflectance" (Section 3.1 3D proposal generation and keypoints sampling)
- "confidence prediction and bounding-box regression" (Section 3. Methodology)
- "two separate FFNs for confidence prediction and bounding box refinement" (Section 3.4 Detect head and training objectives)
- "3 072 raw points are randomly sampled by FPS." (Section 4.2.2 Training and inference details)
- "top-100 proposals are selected for the final prediction." (Section 4.2.2 Training and inference details)
- Inference: Attention Dynamic is Static (inferred) because the model uses a fixed keypoint set ("256 internal keypoints are randomly selected for subsequent processing.") rather than runtime selection; State Dynamic is Constructed (inferred) because it builds a "global proposal representation" used for prediction. (Section 4.2.2 Training and inference details; Section 3.3.2 Dual transformer for proposal refinement)

---

## CSV Output (required)
