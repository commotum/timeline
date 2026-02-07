# LIFT: Learning 4D LiDAR Image Fusion Transformer for 3D Object Detection (Not specified in the paper.)
Source: LIFT- Learning 4D LiDAR Image Fusion Transformer for 3D Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | sequential LiDAR point clouds; sequential camera images | 4D (x, y, z, t) (inferred); 3D (x, y, t) (inferred) | Capped | Static (inferred) | Direct (inferred) | 3D object detections (3D bounding boxes) | 3D (x, y, z) | Not specified in the paper. |

## Summary
The paper focuses on 3D object detection in autonomous driving using sequential LiDAR point clouds and camera images as inputs. It frames the inputs as spatiotemporal streams and limits their size with explicit caps on points and pillars, while using windowed self-attention over BEV grids for fusion. The model is an end-to-end single-stage detector with no explicit persistent state described beyond the input processing pipeline.

## Evidence
### Task: 3D object detection
- "LiDAR and camera are two common sensors to collect data in time for 3D object detection under the autonomous driving context." (Abstract)
- "an end-to-end single-stage 3D object detection approach, which takes both sequential point clouds and images as input." (Section 3)
- "point clouds can be presented as a sequence of frames" (Section 3.1)
- "camera images are presented in time stream" (Section 3.1)
- "LIFT learns to align the input 4D sequential cross-sensor data" (Abstract)
- "we constrain the local self-attention computation within partitioned windows" (Section 3.2)
- "3D bounding boxes are annotated at 2 Hz" (Section 4.1)
- "We limit the max number of points within each pillar to 20." (Section 4.1)
- "limit the max number of non-empty pillars to 30000." (Section 4.1)
- Inference: Input dimensions labeled as 4D (x, y, z, t) and 3D (x, y, t) because the paper describes sequential point clouds, time-stream images, and "4D sequential cross-sensor data." Attention Dynamic set to Static because attention is computed within "partitioned windows"; State Dynamic set to Direct because the model is an "end-to-end single-stage" detector without explicit persistent memory.
