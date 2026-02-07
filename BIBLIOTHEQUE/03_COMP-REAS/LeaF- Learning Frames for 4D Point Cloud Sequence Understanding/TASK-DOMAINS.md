# LeaF: Learning Frames for 4D Point Cloud Sequence Understanding (Not specified in the paper.)
Source: LeaF- Learning Frames for 4D Point Cloud Sequence Understanding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action segmentation (HOI4D) | 4D point cloud sequence | 4D (x, y, z, t) | Fixed | Static (inferred) | Direct (inferred) | action labels per timestamp | 1D (t) (inferred) | Fixed (inferred) |
| action recognition (MSR-Action3D) | 4D point cloud sequence | 4D (x, y, z, t) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action category label (inferred) | 0D (inferred) | Fixed (inferred) |
| indoor semantic segmentation (HOI4D) | 4D point cloud sequence | 4D (x, y, z, t) | Fixed | Static (inferred) | Direct (inferred) | semantic labels per point (inferred) | 4D (x, y, z, t) (inferred) | Fixed (inferred) |
| outdoor semantic segmentation (Synthia4D) | 4D point cloud sequence | 4D (x, y, z, t) | Fixed (inferred) | Static (inferred) | Direct (inferred) | semantic labels per point (inferred) | 4D (x, y, z, t) (inferred) | Fixed (inferred) |

## Summary
LeaF is evaluated on four 4D point cloud sequence tasks: action segmentation, action recognition, and indoor/outdoor semantic segmentation. Inputs are 4D point cloud sequences (3D space plus time) with fixed clip sizes per dataset/experiment (inferred), while outputs range from per-timestamp action labels (1D over time) and sequence-level action labels (0D) to per-point semantic labels over the 4D space-time domain. The model uses frame-guided 4D convolutions and a self-attention fusion with task heads, so attention is treated as static and state as direct (inferred).

## Evidence
### Task: action segmentation (HOI4D)
- "we first conducted experiments on the HOI4D action segmentation task. For each point cloud sequence, we need to predict the action labels for each timestamp." (Section 4.1)
- "Each sequence has 150 timestamps with 2048 points per timestamp." (Section 4.1)
- "point cloud sequences in 4D (3D space + 1D time)." (Section 1 Introduction)
- Inference: Marked Out Dimension/Out Dynamics as 1D (t)/Fixed and Attention/State as Static/Direct because labels are predicted "for each timestamp" in fixed-length sequences and the method uses self-attention with task heads: "We formulate the fusion process as a self-attention operation" (Section 3.3) and "we can add different task heads to complete various 4D point cloud sequence understanding tasks." (Section 3.3)

### Task: action recognition (MSR-Action3D)
- "we used the MAR-Action3D dataset, which consists of 567 human point cloud sequences, including 20 action categories." (Section 4.2)
- "During training, video-level labels are used as segment-level labels." (Section 4.2)
- "point cloud sequences in 4D (3D space + 1D time)." (Section 1 Introduction)
- Inference: Marked Input/Output dynamics as Fixed and output as a single action label (0D) because the task uses fixed clip lengths (e.g., "when the clip length is 8" and "when clip length is 24") and predicts video-level labels; attention/state are inferred Static/Direct based on the self-attention fusion and task-head pipeline: "We formulate the fusion process as a self-attention operation" (Section 3.3) and "we can add different task heads to complete various 4D point cloud sequence understanding tasks." (Section 3.3)

### Task: indoor semantic segmentation (HOI4D)
- "we conducted further experiments on HOI4D for 4D semantic segmentation." (Section 4.3)
- "The dataset consists of 3863 4D sequences, each including 300 timestamps of point clouds." (Section 4.3)
- "For one timestamp, there are 8192 points." (Section 4.3)
- Inference: Marked output as per-point semantic labels over 4D space-time and Out Dynamics as Fixed because the task is semantic segmentation on fixed-size point cloud sequences; attention/state are inferred Static/Direct from the self-attention fusion and task-head design: "We formulate the fusion process as a self-attention operation" (Section 3.3) and "we can add different task heads to complete various 4D point cloud sequence understanding tasks." (Section 3.3)

### Task: outdoor semantic segmentation (Synthia4D)
- "Outdoor Semantic Segmentation on Synthia4D" (Section 4.4)
- "It consists of six sequences of driving scenarios where both objects and cameras are moving." (Section 4.4)
- "other settings are the same as the experiments on semantic segmentation on HOI4D." (Section 4.4)
- Inference: Marked input/output dynamics as Fixed and output as per-point semantic labels over 4D space-time because the outdoor setup reuses the HOI4D semantic segmentation settings; attention/state are inferred Static/Direct from the self-attention fusion and task-head design: "We formulate the fusion process as a self-attention operation" (Section 3.3) and "we can add different task heads to complete various 4D point cloud sequence understanding tasks." (Section 3.3)
