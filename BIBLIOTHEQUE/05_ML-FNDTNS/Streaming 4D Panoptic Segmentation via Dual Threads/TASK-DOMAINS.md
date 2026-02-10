# 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads (Not specified in the paper)
Source: Streaming 4D Panoptic Segmentation via Dual Threads.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Streaming 4D panoptic segmentation | Streaming sequence of point clouds | 4D (x, y, z, t) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Panoptic segmentation on each incoming frame (per-point predictions) | 4D (x, y, z, t) | Open (inferred) |
| Future ego-pose prediction | Ego-pose history from key frames / relative poses | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Relative future ego-pose motion (m frames ahead) | 1D (t) (inferred) | Capped (inferred) |
| Future flow prediction | Key flows between keyframes | 4D (x, y, z, t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Forecast future forward flows for dynamic objects | 4D (x, y, z, t) (inferred) | Capped (inferred) |

## Summary
The paper’s primary model-facing objective is streaming 4D panoptic segmentation over point-cloud sequences, with auxiliary predictive tasks for future ego-pose and future flow to support alignment. Inputs and outputs are centered on spatiotemporal point-cloud domains, plus temporal pose/flow prediction streams used by the dual-thread system. The justified task space spans Open streaming dynamics for the main segmentation pipeline and Capped future-horizon prediction for pose/flow forecasting, with Dynamic memory querying in segmentation and Constructed state via maintained memories/LSTM modules.

## Evidence
### Task: Streaming 4D panoptic segmentation
- "Given a streaming sequence of point clouds, the goal is to predict panoptic segmentation on each frame within a strict time budget, enabling real-time scene perception." (Section 1. Introduction)
- "We propose a new task of streaming 4D panoptic segmentation." (Section 3. Streaming 4D Panoptic Segmentation)
- "These retrieved features are subsequently passed through a lightweight prediction head to produce the final output." (Section 4.1. Dual-thread system)
- Inference: `Open` was inferred from "streaming sequence" and "each incoming frame" wording; `Dynamic` attention and `Constructed` state were inferred from runtime feature retrieval from geometric memory and continuously updated geometry/motion memories (Sections 1, 4.1, 4.2).

### Task: Future ego-pose prediction
- "we utilize ego-pose forecasting to compensate for camera motion and align the current memory with future frames." (Section 4.3. Ego-pose Future Alignment)
- "In order to forecast the relative pose m frames ahead for the future frame  $x_{t+m}$  using pose forecaster F, we have:" (Section 4.3. Ego-pose Future Alignment)
- "the ego-pose forecaster is designed in a multi-head structure, with each head predicting the future pose for a fixed number of frames ahead." (Section 4.3. Ego-pose Future Alignment)
- Inference: `1D (t)` and `Open` input dynamics were inferred because this module forecasts from temporal ego-motion history; `Capped` output dynamics were inferred from the fixed-number future heads; `Static` attention and `Constructed` state were inferred from fixed forecaster inputs with maintained ego-pose memory via LSTM (Section 4.3).

### Task: Future flow prediction
- "we introduce the Future Flow Forecasting in the predictive thread and the Inverse Forward Flow in the inference thread." (Section 4.4. Dynamic Object Future Alignment)
- "These key flows are then input into the LSTM [15] to forecast future flows, supporting the fast alignment of dynamic objects across memory and incoming frames." (Section 4.4. Dynamic Object Future Alignment)
- "The remaining components, including ego-pose forecasting, forward flow forecasting, and history memory aggregation, are trained subsequently." (Section 5.1. Settings)
- Inference: `4D (x, y, z, t)` and `Open` input dynamics were inferred because key flows come from streaming keyframes in point-cloud sequences; `Capped` output dynamics were inferred from multi-frame-ahead forecasting intent; `Static` attention and `Constructed` state were inferred from LSTM-based forecasting over maintained temporal memory (Sections 4.4, 5.1).
