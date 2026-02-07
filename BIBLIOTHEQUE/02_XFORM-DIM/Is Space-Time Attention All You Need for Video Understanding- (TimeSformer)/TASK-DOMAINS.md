# Is Space-Time Attention All You Need for Video Understanding? (Not specified in the paper.)
Source: Is Space-Time Attention All You Need for Video Understanding- (TimeSformer).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Video classification (action recognition) | RGB video clip (frames) | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Video class label | 0D (inferred) | Fixed (inferred) |
| Long-term task classification (instructional videos) | Long instructional video (minutes) | 3D (x, y, t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Long-term task label | 0D (inferred) | Fixed (inferred) |

## Summary
TimeSformer is evaluated on video classification tasks, covering action recognition benchmarks and long-term task classification on instructional videos. Inputs are RGB video clips or long videos (spatiotemporal), and outputs are single task/action labels. The paper describes fixed-size clips for standard action recognition and variable-length long videos handled by aggregating many clips; attention is static and state is direct (both inferred from the clip-based, feedforward classification setup).

## Evidence
### Task: Video classification (action recognition)
- "We present a convolution-free approach to video classification built exclusively on self-attention over space and time." (Abstract)
- "We evaluate TimeSformer on four popular action recognition datasets." (Section 4. Experiments)
- "The TimeSformer takes as input a clip" (Section 3. The TimeSformer Model)
- "consisting of F RGB frames" (Section 3. The TimeSformer Model)
- "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (Section 3. The TimeSformer Model)
- "Unless differently indicated, we use clips of size  8 × 224 × 224" (Section 4. Experiments)
- "Kinetics-400 (Carreira & Zisserman, 2017) consists of 240K training videos and 20K validation videos that span 400 human action categories." (Appendix: Datasets)
- Inference: In Dimension = 3D (x, y, t) and In Dynamics = Fixed are inferred from the clip-based RGB frame input and fixed clip size ("takes as input a clip", "consisting of F RGB frames", "clips of size  8 × 224 × 224"). Attention Dynamic = Static and State Dynamic = Direct are inferred because the model processes the full clip and maps it to labels via an MLP without any external state ("predict the final video classes"). Out Dimension = 0D and Out Dynamics = Fixed are inferred from the single-label class output and fixed category sets ("video classes", "span 400 human action categories").

### Task: Long-term task classification (instructional videos)
- "Lastly, we evaluate TimeSformer on the task of long-term video modeling using HowTo100M." (Section 4.6. Long-Term Video Modeling)
- "HowTo100M is an instructional video dataset that contains around 1M instructional Web videos" (Section 4.6. Long-Term Video Modeling)
- "Given a video spanning several minutes, the goal is to predict the long-term task demonstrated in the video" (Table 8 caption)
- "This gives a subset of HowTo100M corresponding to 120K videos spanning 1059 task categories." (Section 4.6. Long-Term Video Modeling)
- "we sample as many non-overlapping temporal clips as needed to cover the full temporal extent of a video" (Section 4.6. Long-Term Video Modeling)
- Inference: In Dimension = 3D (x, y, t) is inferred from the video input ("instructional video dataset", "video spanning several minutes"). In Dynamics = Open is inferred because the method covers the full temporal extent by sampling as many clips as needed, implying variable-length inputs. Attention Dynamic = Static and State Dynamic = Direct are inferred from the same clip-based, feedforward classification setup used for TimeSformer ("predict the long-term task"). Out Dimension = 0D and Out Dynamics = Fixed are inferred from the single-label task prediction and the fixed set of task categories ("1059 task categories").
