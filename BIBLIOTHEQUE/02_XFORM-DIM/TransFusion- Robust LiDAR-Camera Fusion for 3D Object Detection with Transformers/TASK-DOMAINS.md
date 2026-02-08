# TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers (Year not specified in the paper)
Source: TransFusion- Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | LiDAR point clouds; multi-view camera images | 3D (x, y, z); 2D (x, y) | Capped (inferred) | Dynamic | Constructed (inferred) | 3D bounding boxes; class labels | 3D (x, y, z); 0D | Capped |
| 3D multi-object tracking (MOT) | Per-frame 3D detections from LiDAR-camera observations over time (inferred) | 4D (x, y, z, t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | 3D object tracks with identities over time (inferred) | 4D (x, y, z, t) (inferred) | Open (inferred) |

## Summary
The paper explicitly covers LiDAR-camera 3D object detection and also extends evaluation to 3D multi-object tracking via tracking-by-detection. Detection combines 3D point-cloud and 2D image inputs, uses dynamic attention, and outputs capped sets of 3D boxes plus class labels. The tracking extension indicates temporal 4D task structure and constructed tracking state by inference from the MOT framing, while tracking-specific attention behavior is not specified in the OCR text.

## Evidence
### Task: 3D object detection
- "As one of the fundamental tasks in self-driving, 3D object detection aims to localize a set of objects in 3D space and recognize their categories." (Section 1. Introduction)
- "given a LiDAR BEV feature map and an image feature map from convolutional backbones, our transformer-based detection head first decodes object queries into initial bounding box predictions using the LiDAR information, and then performs LiDAR-camera fusion by attentively fusing object queries with useful image features." (Section 3. Methodology)
- "The attention mechanism of the transformer enables our model to adaptively determine where and what information should be taken from the image" (Abstract)
- "Then we regard the heatmap as  $X \times Y \times K$  object candidates and select the top-N candidates for all the categories as our initial object queries." (Section 3.2. Query Initialization)
- "By decoding each object query into prediction in parallel, we get a set of predictions  $\{\hat{b}_t, \hat{p}_t\}_t^N$  as output" (Section 3.3. Transformer Decoder and FFN)
- Inference: `In Dynamics = Capped (inferred)` because inputs are processed with fixed-size image/backbone features and top-N query selection; `State Dynamic = Constructed (inferred)` because the first decoder layer creates initial box/query state that is reused by the fusion decoder (Sections 3.2, 3.3, 3.4).

### Task: 3D multi-object tracking (MOT)
- "We also extend the proposed method to the 3D tracking task and achieve the 1st place in the leaderboard of nuScenes tracking" (Abstract)
- "we evaluate our model in a 3D multi-object tracking (MOT) task by performing tracking-by-detection with the same tracking algorithms adopted by CenterPoint." (Section 5.1. Extend to Tracking)
- "The nuScenes dataset is a large-scale autonomous-driving dataset for 3D detection and tracking" (Section 5. Experiments, nuScenes Dataset)
- Inference: `Input/Output` are treated as temporal tracking entities, so `In/Out Dimension = 4D (x, y, z, t) (inferred)` and `In/Out Dynamics = Open (inferred)` from the MOT and tracking-by-detection framing; `State Dynamic = Constructed (inferred)` because tracking-by-detection implies maintaining associations/identities across time (Abstract; Section 5.1. Extend to Tracking).
