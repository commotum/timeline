# VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding (2021)
Source: VideoCLIP- Contrastive Pretraining for Zero-Shot Video-Text Understanding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text-to-video retrieval | Text queries and video clips | 1D (t); 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Ranked video clips for each text query (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Multiple-choice VideoQA | Video clips with candidate textual answers | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | One selected textual answer | 0D (inferred) | Capped (inferred) |
| Action segmentation | Video tokens/frames with predefined segment text labels | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Per-token/frame segment labels with Outside rejection | 1D (t) | Capped (inferred) |
| Action step localization | Video tokens/frames with task-specific step text descriptions | 3D (x, y, t); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Per-token/frame step assignment distribution | 1D (t) | Capped (inferred) |

## Summary
VideoCLIP explicitly targets four downstream tasks: text-to-video retrieval, multiple-choice VideoQA, action segmentation, and action step localization. The paper covers multimodal inputs combining text sequences and video, supporting both sequence-level matching and token/frame-level labeling. Based on the OCR description of token/frame limits and windowed inference, the task interfaces are best characterized as Capped dynamics, with mostly Static attention and Direct state usage (inferred). Dimension coverage spans 0D outputs for answer selection and 1D temporal outputs for token/frame assignments, with video inputs grounded in 3D (x, y, t) and text in 1D (t).

## Evidence
### Task: Text-to-video retrieval
- "After pre-training, we apply our model for zeroshot transfer *without* any fine-tuning on target dataset labels. We directly use our pre-trained model on a diverse set of *four* tasks in *five* datasets, including text-video retrieval (for text-to-video similarity)..." (Section 1 Introduction)
- "**Text** $\rightarrow$ **Video Retrieval.** Text $\rightarrow$ video retrieval tests the text-to-video similarity computed on the learned video-text representation." (Section 4 Zero-shot Transfer to End Tasks)
- "We directly use our video and text Transformers to encode the videos and the text queries and measure the text-to-video similarities for retrieval." (Section A.1 End Task Setup Details)
- Inference: 1D (t) text and 3D (x, y, t) video dimensions are inferred from the text-query/video-clip formulation; Capped dynamics are inferred from "We limit the maximum number of video tokens to be 32... For text transformer, we have 61 text tokens..." (Section 5.3 Implementation Details). Static attention and Direct state are inferred because the task is described as direct encoding and similarity measurement without runtime input-selection or persistent constructed memory.

### Task: Multiple-choice VideoQA
- "Multiple-choice VideoQA. In multiple-choice VideoQA (Yu et al., 2018), the model aligns each video with one out of several text candidate answers." (Section 4 Zero-shot Transfer to End Tasks)
- "We formulate this task as ranking candidate textual answers for a given video question query." (Section 4 Zero-shot Transfer to End Tasks)
- "Recall that this task can be formulated as a video-text retrieval task except the candidate textual answers are associated with each video and only one answer is correct (most relevant)." (Section 5.2 End Task Setups)
- Inference: Input dimensions (video + text) and Capped dynamics are inferred from multimodal clips plus finite candidate answers ("On average, VideoQA for MSR-VTT has 5 candidate answers per video," Section 5.2) and token limits in Section 5.3. Output 0D is inferred from "one ... correct" selected answer. Static attention and Direct state are inferred from similarity ranking over provided candidates.

### Task: Action segmentation
- "Action segmentation assigns each token (or frame) of a video with one of the pre-defined labels to separate meaningful segments of videos from the rest tokens (or frames)." (Section 4 Zero-shot Transfer to End Tasks)
- "As such, the hidden state of each video token can have a distribution of similarity over segment labels." (Section 4 Zero-shot Transfer to End Tasks)
- "There are 778 segment labels... we do not model the Outside label explicitly and determine an Outside label only when all other 778 labels reject a video token." (Section 5.2 End Task Setups)
- Inference: 3D (x, y, t) video and 1D (t) text-label dimensions are inferred from frame/token labeling with textual labels. Capped dynamics are inferred from finite label set and windowed processing: "we apply a sliding window with a step size of 16 seconds and a window size of 32 seconds" (Section 5.2), together with Section 5.3 token caps. Static attention and Direct state are inferred from per-token similarity scoring over provided labels.

### Task: Action step localization
- "Action step localization is to assign each video token to one or multiple steps in the associated task." (Section 4 Zero-shot Transfer to End Tasks)
- "Then we separately forward text labels into the text backbone to obtain the hidden states of step labels $z_S$. The distribution of each video token over steps is predicted as Softmax $(h_u z_S^T)$." (Section 4 Zero-shot Transfer to End Tasks)
- "Each task has a set of steps in the form of text descriptions and each frame of video is annotated with one or multiple steps as a distribution." (Section 5.2 End Task Setups)
- Inference: Input dimensions are inferred as video (3D (x, y, t)) plus text step descriptions (1D (t)). Capped dynamics are inferred from finite per-task step sets and model token limits (Section 5.3). Static attention and Direct state are inferred because inference computes token-to-step similarities over given inputs without adaptive retrieval or persistent constructed state.
