## 1. Basic Metadata

- Title: "Video Swin Transformer" (Title)
- Authors: "Ze Liu\*12, Jia Ning\*13, Yue Cao¹†, Yixuan Wei¹⁴, Zheng Zhang¹, Stephen Lin¹, Han Hu¹†" (Title page)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes Video Swin Transformer, "a pure-transformer backbone architecture for video recognition" that leverages "spatiotemporal locality of videos" to improve the speed-accuracy trade-off for video recognition. (Section 1 Introduction)

## 3. Tasks Evaluated

- Task name: Action recognition
  - Task type: Classification
  - Dataset(s) used: Kinetics-400 (K400), Kinetics-600 (K600)
  - Domain: Video
  - Evidence: "For human action recognition, we adopt two versions of the widely-used Kinetics [20] dataset, Kinetics-400 and Kinetics-600." (Section 4.1 Setup)
  - Evidence: "Kinetics-400 (K400) consists of  $\sim$ 240k training videos and 20k validation videos in 400 human action categories." (Section 4.1 Setup)
  - Evidence: "Kinetics-600 (K600) is an extension of K400 that contains  $\sim$ 370k training videos and 28.3k validation videos from 600 human action categories." (Section 4.1 Setup)
  - Evidence: "The proposed approach shows strong performance on the video recognition tasks of action recognition on Kinetics-400/Kinetics-600" (Section 1 Introduction)

- Task name: Temporal modeling
  - Task type: Classification
  - Dataset(s) used: Something-Something V2 (SSv2)
  - Domain: Video
  - Evidence: "For temporal modeling, we utilize the popular Something-Something V2 (SSv2) [14] dataset, which consists of 168.9K training videos and 24.7K validation videos over 174 classes." (Section 4.1 Setup)
  - Evidence: "The proposed approach shows strong performance on the video recognition tasks of action recognition on Kinetics-400/Kinetics-600 and temporal modeling on Something-Something v2 (abbreviated as SSv2)." (Section 1 Introduction)

## 4. Domain and Modality Scope

- Evaluation domain scope: Single domain (video). Evidence: "The proposed approach shows strong performance on the video recognition tasks of action recognition on Kinetics-400/Kinetics-600 and temporal modeling on Something-Something v2 (abbreviated as SSv2)." (Section 1 Introduction)
- Multiple domains within the same modality: Not stated; datasets cited are all video. Evidence: "The input video is defined to be of size  $T \times H \times W \times 3$" (Section 3.1 Overall Architecture)
- Multiple modalities: Not stated. Evidence: "The input video is defined to be of size  $T \times H \times W \times 3$" (Section 3.1 Overall Architecture)
- Domain generalization or cross-domain transfer: Not claimed. Evidence: "As our architecture is adapted from Swin Transformer, it can readily be initialized with a strong model pre-trained on a large-scale image dataset." (Section 1 Introduction)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Action recognition (Kinetics-400/Kinetics-600) | Not stated. | Yes; initialized from pre-trained model and trained on K400/K600. | Yes; head is randomly initialized for K400/K600 training. | "For K400 and K600, we employ an AdamW [21] optimizer for 30 epochs" (Section 4.1 Setup); "As the backbone is initialized from the pre-trained model but the head is randomly initialized" (Section 4.1 Setup) |
| Temporal modeling (SSv2) | Not stated. | Yes; initialized from Kinetics-400 model. | Not stated. | "For temporal modeling, we utilize the popular Something-Something V2 (SSv2) [14] dataset" (Section 4.1 Setup); "As also done in [9], we use the model pre-trained on Kinetics-400 as initialization" (Section 4.1 Setup) |

## 6. Input and Representation Constraints

- Input format and dimensionality: "The input video is defined to be of size  $T \times H \times W \times 3$" (Section 3.1 Overall Architecture)
- Patch/token size and tokenization: "we treat each 3D patch of size  $2 \times 4 \times 4 \times 3$  as a token." (Section 3.1 Overall Architecture)
- Token count and feature size: "the 3D patch partitioning layer obtains  $\frac{T}{2} \times \frac{H}{4} \times \frac{W}{4}$  3D tokens, with each patch/token consisting of a 96-dimensional feature." (Section 3.1 Overall Architecture)
- Temporal downsampling assumption: "we do not down-sample along the temporal dimension." (Section 3.1 Overall Architecture)
- Spatial downsampling scheme: "performs  $2\times$  spatial downsampling in the patch merging layer of each stage." (Section 3.1 Overall Architecture)
- Default window size: "The window size is set to P=8 and M=7 by default." (Section 3.3 Architecture Variants)
- Training-time clip length and resolution: "we sample a clip of 32 frames from each full length video using a temporal stride of 2 and spatial size of  $224 \times 224$ , resulting in  $16 \times 56 \times 56$  input 3D tokens." (Section 4.1 Setup)
- Inference resizing/cropping: "the shorter spatial side is scaled to 224 pixels and we take 3 crops of size  $224 \times 224$  that cover the longer spatial axis." (Section 4.1 Setup)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified. Evidence of experiment-specific token count: "we sample a clip of 32 frames from each full length video using a temporal stride of 2 and spatial size of  $224 \times 224$ , resulting in  $16 \times 56 \times 56$  input 3D tokens." (Section 4.1 Setup)
- Fixed or variable sequence length: General input is defined by T, H, W; experiments sample 32-frame clips. Evidence: "The input video is defined to be of size  $T \times H \times W \times 3$" (Section 3.1 Overall Architecture); "we sample a clip of 32 frames from each full length video" (Section 4.1 Setup)
- Attention type: Windowed + shifted + hierarchical. Evidence: "Multi-head self-attention on non-overlapping 3D windows" (Section 3.2 3D Shifted Window based MSA Module); "we extend the shifted 2D window mechanism of Swin Transformer to 3D windows for the purpose of introducing cross-window connections while maintaining the efficient computation of non-overlapping window based self-attention." (Section 3.2 3D Shifted Window based MSA Module); "strictly follow the hierarchical architecture of the original Swin Transformer [28], which consists of four stages and performs  $2\times$  spatial downsampling in the patch merging layer of each stage." (Section 3.1 Overall Architecture)
- Computational cost mechanisms: "full spatiotemporal self-attention can be well-approximated by self-attention computed locally, at a significant saving in computation and model size." (Section 1 Introduction); "the windows are arranged to evenly partition the video input in a non-overlapping manner." (Section 3.2 3D Shifted Window based MSA Module)

## 8. Positional Encoding (Critical Section)

- Mechanism: Relative, bias-based. Evidence: "we follow [28] by introducing 3D relative position bias  $B \in \mathbb{R}^{P^2 \times M^2 \times M^2}$  for each head" (Section 3.2 3D Shifted Window based MSA Module)
- Where applied: Attention bias in windowed MSA. Evidence: "$$(Q, K, V) = \text{SoftMax}(QK^T/\sqrt{d} + B)V,$$" (Section 3.2 3D Shifted Window based MSA Module)
- Fixed across experiments or modified per task: Fixed mechanism; only initialization is varied. Evidence: "For the 3D relative position bias matrix, we also have two different initialization choices, duplicate or center initialization." (Section 4.3 Ablation Study)
- Ablated/compared alternatives: Only initialization variants for the relative position bias are compared. Evidence: "Table 9: Ablation study on the two initialization methods of 3D relative position bias matrix with Swin-T on K400." (Section 4.3 Ablation Study)

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Fixed architectural assumption with limited ablation on initialization. Evidence: "For the 3D relative position bias matrix, we also have two different initialization choices, duplicate or center initialization." (Section 4.3 Ablation Study)
- Multiple positional encodings compared: Not stated; only initialization choices are compared. Evidence: "we initialize the 3D relative position bias matrix by masking the relative position bias across different frames with a small negative value (e.g. -4.6), so that each token only focuses inside the same frame from the very beginning." (Section 4.3 Ablation Study)
- Claims that PE choice is not critical or secondary: Not stated. Evidence: "As shown in Table 9, we find that both initialization methods achieve the same top-1 accuracy of 78.8% with Swin-T on K400." (Section 4.3 Ablation Study)

## 10. Evidence of Constraint Masking

- Model sizes: "200.0M params for Swin-L vs. 647.5M params for ViViT-H" (Section 1 Introduction)
- Dataset sizes: "Kinetics-400 (K400) consists of  $\sim$ 240k training videos and 20k validation videos in 400 human action categories."; "Kinetics-600 (K600) is an extension of K400 that contains  $\sim$ 370k training videos and 28.3k validation videos from 600 human action categories."; "Something-Something V2 (SSv2) [14] dataset, which consists of 168.9K training videos and 24.7K validation videos over 174 classes." (Section 4.1 Setup)
- Scaling vs architecture attribution: "we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization." (Abstract); "full spatiotemporal self-attention can be well-approximated by self-attention computed locally, at a significant saving in computation and model size." (Section 1 Introduction)
- Data/model scaling claims: "action recognition (84.9 top-1 accuracy on Kinetics-400 and 86.1 top-1 accuracy on Kinetics-600 with  $\sim 20 \times$  less pre-training data and  $\sim 3 \times$  smaller model size)" (Abstract)
- Training tricks: "we find that multiplying the backbone learning rate by 0.1 improves performance (shown in Tab. 7)." (Section 4.1 Setup); "we observe that a lower learning rate of the backbone architecture (e.g.  $0.1\times$ ) relative to that of the head, which is randomly initialized, brings gains in top-1 accuracy for K400." (Section 4.3 Ablation Study)

## 11. Architectural Workarounds

- Windowed attention for efficiency: "the windows are arranged to evenly partition the video input in a non-overlapping manner." (Section 3.2 3D Shifted Window based MSA Module)
- Shifted windows for cross-window connections: "we extend the shifted 2D window mechanism of Swin Transformer to 3D windows for the purpose of introducing cross-window connections while maintaining the efficient computation of non-overlapping window based self-attention." (Section 3.2 3D Shifted Window based MSA Module)
- Hierarchical stages and patch merging: "strictly follow the hierarchical architecture of the original Swin Transformer [28], which consists of four stages and performs  $2\times$  spatial downsampling in the patch merging layer of each stage." (Section 3.1 Overall Architecture)
- Fixed 3D patch tokens: "we treat each 3D patch of size  $2 \times 4 \times 4 \times 3$  as a token." (Section 3.1 Overall Architecture)

## 12. Explicit Limitations and Non-Claims

- Future work: "Our approach could be further improved via using larger model (e.g. Swin-L), larger resolution of input (e.g. 384<sup>2</sup>) and better pre-trained model (e.g. K600). We leave these attempts as future work." (Section 4.2 Comparison to state-of-the-art)
- Pending analysis: "As this observation is inconsistent with that in [1], we will analyze the difference once the code of ViViT is released." (Section 4.3 Ablation Study)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single modality/domain (video) with evaluations on Kinetics-400/600 and SSv2.
> – Task structure: Two supervised classification tasks (action recognition, temporal modeling) on video benchmarks.
> – Representation rigidity: Fixed 3D patch tokens (2 x 4 x 4), fixed window sizes (P=8, M=7), and fixed training clip length (32 frames at 224 x 224) in reported experiments.
> – Model sharing vs specialization: Pretrained backbones are reused (ImageNet or K400), and shared weights across tasks are not stated.
> – Role of positional encoding: 3D relative position bias is a fixed attention bias with only initialization variants ablated.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates two tasks in the same modality/domain of video: "action recognition on Kinetics-400/Kinetics-600" and "temporal modeling on Something-Something v2" (Section 1 Introduction). All datasets are video and the input is consistently described as video tensors: "The input video is defined to be of size  $T \times H \times W \times 3$" (Section 3.1 Overall Architecture).
