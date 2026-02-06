# DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving (Not specified in the paper.)
Source: DriveWorld- 4D Pre-trained Scene Understanding via World Models for Autonomous Driving.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D occupancy prediction (current and future) | multi-camera video frames; expert actions | 3D (x, y, t) (inferred); 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | 3D occupancy (current and future time steps) | 4D (x, y, z, t) (inferred) | Fixed (inferred) |
| Action prediction (velocity, steering) | multi-camera video frames; expert actions | 3D (x, y, t) (inferred); 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | actions (velocity, steering) | 1D (t) (inferred) | Fixed (inferred) |
| Occupancy flow prediction | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | occupancy flow | Not specified in the paper. | Not specified in the paper. |
| 3D object detection | multi-camera images (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | 3D object detections (inferred) | 3D (x, y, z) (inferred) | Not specified in the paper. |
| Online mapping | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | online map (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Multi-object tracking | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | object tracks over time (inferred) | 4D (x, y, z, t) (inferred) | Not specified in the paper. |
| Motion forecasting | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | future motion trajectories (inferred) | 3D (x, y, t) (inferred) | Not specified in the paper. |
| Occupancy prediction (2D BEV) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | occupancy in 2D BEV view | 2D (x, y) (inferred) | Not specified in the paper. |
| Planning | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | planned trajectory/actions (inferred) | 3D (x, y, t) (inferred) | Not specified in the paper. |

## Summary
DriveWorld pre-trains a world model to predict current/future 3D occupancy and actions from multi-camera video sequences, and is evaluated on downstream autonomous driving tasks including 3D object detection, online mapping, multi-object tracking, motion forecasting, occupancy prediction, and planning. The task coverage spans spatiotemporal inputs and outputs, with inferred dimensions ranging from 2D BEV maps/occupancy to 3D/4D scene representations and motion trajectories. Only the pre-training setup specifies fixed-length sequences (T=4, L=4); attention and state dynamics are not explicitly specified.

## Evidence
### Task: 3D occupancy prediction (current and future)
- "predicts current and future 3D occupancy given the past multi-camera images and actions." (Section 3)
- "The model observes inputs over T=4 steps, and the future prediction is set at L=4 steps." (Section 4.1)
- Inference: In/Out Dimension and Fixed dynamics inferred from the quoted statements about multi-camera images/actions over time steps and predicting current/future 3D occupancy. (Sections 3, 4.1)

### Task: Action prediction (velocity, steering)
- "a Decoder to predict the actions and 3D occupancy" (Section 3)
- "we utilize MLP for action prediction, including velocity and steering." (Section 3.1)
- "The model observes inputs over T=4 steps, and the future prediction is set at L=4 steps." (Section 4.1)
- Inference: In Dimension and Out Dimension (temporal sequences) and Fixed dynamics inferred from the time-step setup and action prediction description. (Sections 3, 3.1, 4.1)

### Task: Occupancy flow prediction
- "we also utilize an L2 loss for occupancy flow prediction." (Section 3.4)

### Task: 3D object detection
- "the multi-camera 3D object detection task." (Section 4.3)
- Inference: Input as multi-camera images and output/dimension as 3D detections inferred from the task description. (Section 4.3)

### Task: Online mapping
- "We validate the performance on the online mapping task." (Section 4.3)
- Inference: Output as a map and 2D dimension inferred from the task description. (Section 4.3)

### Task: Multi-object tracking
- "We further evaluate the performance on the multi-object tracking task" (Section 4.3)
- Inference: Output as object tracks over time and 4D dimension inferred from the task description. (Section 4.3)

### Task: Motion forecasting
- "Motion Forecasting. In the motion prediction task," (Section 4.3)
- "capability to forecast future states" (Section 4.3)
- Inference: Output as future motion trajectories and 3D (x, y, t) dimension inferred from references to forecasting future states. (Section 4.3)

### Task: Occupancy prediction (2D BEV)
- "The UniAD's occupancy prediction task is carried out in the 2D BEV view." (Section 4.3)
- Inference: Out Dimension 2D (x, y) inferred from "2D BEV view." (Section 4.3)

### Task: Planning
- "We finally validate the effectiveness of the proposed 4D pre-training algorithm on the planning task." (Section 4.3)
- "reducing an 0.34m average L2 error" (Section 4.3)
- Inference: Output as planned trajectory/actions and 3D (x, y, t) dimension inferred from the planning task description and L2 trajectory error metric. (Section 4.3)
