# OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving (Not specified in the paper)
Source: OccWorld- Learning a 3D Occupancy World Model for Autonomous Driving.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4D occupancy forecasting | historical 3D occupancy (past frames) | 4D (x, y, z, t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | future 3D occupancy (forecasted frames) | 4D (x, y, z, t) (inferred) | Fixed (inferred) |
| Motion planning | past scenes and ego positions (surrounding information / perception results) (inferred) | 4D (x, y, z, t); 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | future trajectory (series of 2D waypoints in the BEV plane) | 1D (t) (inferred) | Fixed (inferred) |

## Summary
OccWorld is evaluated on two autonomous-driving tasks: 4D occupancy forecasting from past occupancy and motion planning that outputs BEV waypoint trajectories. The occupancy task operates over spatiotemporal 3D occupancy volumes, while planning outputs temporal waypoint sequences; both use fixed 2-second history and 3-second prediction windows (inferred). The model uses masked temporal attention over fixed token sets and constructs discrete scene tokens, indicating static attention and constructed state (both inferred).

## Evidence
### Task: 4D occupancy forecasting
- "we explore 4D occupancy forecasting, which aims to forecast future 3D occupancy given historical occupancy." (Section 4.1 Task Descriptions)
- "predict the 3D occupancy of the following frames given a few past frames." (Section 1 Introduction)
- "We followed existing works [18,26] and used a 2-second historical context to forecast the subsequent 3 seconds." (Section 4.3 Implementation Details)
- Inference: In/Out Dimension are 4D and In/Out Dynamics are Fixed because the task forecasts 3D occupancy across past/future frames with a 2-second history and 3-second future window; Attention Dynamic is Static from "TA denotes masked temporal attention which blocks the effect of future tokens to previous tokens." (Section 3.3 Spatial-Temporal Generative Transformer); State Dynamic is Constructed from "We train a vector-quantized autoencoder (VQ-VAE) [42] on  $\bf y$  to obtain discrete tokens  $\bf z$" (Section 3.2 3D Occupancy Scene Tokenizer).

### Task: Motion planning
- "Motion planning aims to produce safe future trajectories for the vehicle given ground-truth surrounding information or perception results." (Section 4.1 Task Descriptions)
- "The planned trajectory is represented by a series of 2D waypoints in the BEV plane." (Section 4.1 Task Descriptions)
- "a world model w takes as inputs the past scenes and ego positions and predicts their outcome after driving a certain time interval." (Section 3.3 Spatial-Temporal Generative Transformer)
- Inference: Input and In Dimension are marked because planning uses the world model's past scenes and ego positions and the 3D occupancy scene representation: "we propose to adopt 3D occupancy as the 3D scene representation  $\mathbf{y} \in \mathbb{R}^{H \times W \times D}$ ." (Section 3.2 3D Occupancy Scene Tokenizer). Out Dimension is 1D (t) because the output is a "series" of waypoints. In/Out Dynamics are Fixed from "We followed existing works [18,26] and used a 2-second historical context to forecast the subsequent 3 seconds." (Section 4.3 Implementation Details). Attention Dynamic is Static from "TA denotes masked temporal attention which blocks the effect of future tokens to previous tokens." (Section 3.3). State Dynamic is Constructed from "We train a vector-quantized autoencoder (VQ-VAE) [42] on  $\bf y$  to obtain discrete tokens  $\bf z$" (Section 3.2).
