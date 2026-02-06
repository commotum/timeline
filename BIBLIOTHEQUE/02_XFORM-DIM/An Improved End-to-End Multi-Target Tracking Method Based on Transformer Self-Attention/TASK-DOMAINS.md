# An Improved End-to-End Multi-Target Tracking Method Based on Transformer Self-Attention (Not specified in the paper.)
Source: An Improved End-to-End Multi-Target Tracking Method Based on Transformer Self-Attention.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-camera multi-target tracking | Single-camera video frames / detection frames (images) | 3D (x, y, t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Tracking results with IDs across frames (trajectories) | 3D (x, y, t) (inferred) | Not specified in the paper. |
| Cross-camera multi-target tracking / re-identification | Multi-camera video frames / detection frames (images) and raster semantic map | 3D (x, y, t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Cross-camera tracking trajectories with shared IDs (re-identification results) | 3D (x, y, t) (inferred) | Not specified in the paper. |

## Summary
The paper covers multi-target tracking in video, evaluated for single-camera MOT17 and cross-camera OVIT-MOT01 pedestrian tracking/re-identification scenarios. Inputs and outputs are image sequences producing tracked trajectories/IDs, so the task domain is spatiotemporal (3D (x, y, t)) by inference. The paper does not specify interface dynamics or attention dynamics, but it describes constructed internal state via a semantic raster map and a query bank.

## Evidence
### Task: Single-camera multi-target tracking
- "Validation of single camera accuracy results based on the publicly available dataset (MOT17)" (Section 3.2)
- "Single camera field of view based target tracking mechanism" (Section 2.2.3.3)
- "obtain the target tracking results of continuous detection frames in a single camera field of view to achieve uniformity of ID and confidence" (Section 2.2.3.3)
- "the corresponding detection frames and texture features were first obtained based on a multi-dimensional feature extraction CNN network and fed into the encoder" (Section 2.2 Methods)
- "constructed a raster semantic map to encode target locations" (Section 2.2 Methods)
- "a query ensemble  $q_{bank}$  is constructed in combination with a historical continuous frame tracking query" (Section 2.2.3.4)
- Inference: Set In/Out Dimension to 3D (x, y, t) because the paper describes "continuous detection frames"; set State Dynamic to Constructed because it "constructed a raster semantic map" and "a query ensemble  $q_{bank}$  is constructed". (Sections 2.2.3.3, 2.2 Methods, 2.2.3.4)

### Task: Cross-camera multi-target tracking / re-identification
- "video captured by five cameras" (Section 2.1.1)
- "evaluate the accuracy of cross-camera pedestrian re-identification and tracking" (Section 2.1.1)
- "Cross-camera targets were continuously tracked based on three dimensions (i.e., temporal, spatial and logical)" (Section 2.2 Methods)
- "alignment of the multi-camera view tracking trajectory is achieved" (Section 2.2.3.3)
- "The aligned detection results will share the same ID and confidence score" (Section 2.2.3.3)
- "the encoder received the raster semantic map, which was constructed based on the target scene" (Section 2.2 Methods)
- "a query ensemble  $q_{bank}$  is constructed in combination with a historical continuous frame tracking query" (Section 2.2.3.4)
- Inference: Set In/Out Dimension to 3D (x, y, t) because the paper describes "video captured by five cameras" and "continuous" tracking; set State Dynamic to Constructed because it "constructed a raster semantic map" and "a query ensemble  $q_{bank}$  is constructed". (Sections 2.1.1, 2.2 Methods, 2.2.3.4)
