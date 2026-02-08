# Video Swin Transformer (Year not specified in the paper)
Source: Video Swin Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action recognition (classification) | Video clips | 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Action class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Temporal modeling (classification) | Video clips | 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Temporal class label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates two video recognition tasks: action recognition and temporal modeling. Both tasks take video clips indexed in space and time, which maps to 3D (x, y, t), and the reported setup uses bounded clip sampling rather than open-ended streams. Outputs are classification labels (0D) with fixed-size prediction targets per inference. Attention is static because window partitioning/shift is predefined, and state is direct because no persistent constructed memory is described.

## Evidence
### Task: Action recognition (classification)
- "The proposed approach shows strong performance on the video recognition tasks of action recognition on Kinetics-400/Kinetics-600 and temporal modeling on Something-Something v2 (abbreviated as SSv2)." (Section 1 Introduction)
- "For human action recognition, we adopt two versions of the widely-used Kinetics [20] dataset, Kinetics-400 and Kinetics-600." (Section 4.1 Setup)
- "The input video is defined to be of size  $T \times H \times W \times 3$" (Section 3.1 Overall Architecture)
- Inference: `3D (x, y, t)` is inferred from the explicit `T \times H \times W` video indexing; `Capped` is inferred from fixed clip sampling ("we sample a clip of 32 frames ... spatial size of  $224 \times 224$ , resulting in  $16 \times 56 \times 56$  input 3D tokens." in Section 4.1 Setup); `Static` is inferred from fixed windowed attention ("the windows are arranged to evenly partition the video input in a non-overlapping manner." in Section 3.2 3D Shifted Window based MSA Module); `Direct` is inferred because the paper describes feed-forward/video-transformer mapping without persistent external state; output `0D` fixed class labels are inferred from recognition protocol and category sets ("For all methods, we follow prior art by reporting top-1 and top-5 recognition accuracy." in Section 4.1 Setup; "...400 human action categories." in Section 4.1 Setup).

### Task: Temporal modeling (classification)
- "The proposed approach shows strong performance on the video recognition tasks of action recognition on Kinetics-400/Kinetics-600 and temporal modeling on Something-Something v2 (abbreviated as SSv2)." (Section 1 Introduction)
- "For temporal modeling, we utilize the popular Something-Something V2 (SSv2) [14] dataset, which consists of 168.9K training videos and 24.7K validation videos over 174 classes." (Section 4.1 Setup)
- "For SSv2, we employ an AdamW [21] optimizer for longer training of 60 epochs with 2.5 epochs of linear warm-up." (Section 4.1 Setup)
- Inference: Input dimensionality, attention behavior, and state assignment are inferred identically to action recognition from the same architecture statements (Section 3.1 and Section 3.2). `Capped` is inferred from bounded clip/window configuration used in evaluation setup, and output is inferred as fixed `0D` class labels from the stated classification metric and class taxonomy ("For all methods, we follow prior art by reporting top-1 and top-5 recognition accuracy." and "...174 classes." in Section 4.1 Setup).
