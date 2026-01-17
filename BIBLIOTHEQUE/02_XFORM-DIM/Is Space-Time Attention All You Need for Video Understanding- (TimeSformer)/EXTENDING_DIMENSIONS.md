## 1. Basic Metadata
- Title: "Is Space-Time Attention All You Need for Video Understanding?" (Title)
- Authors: "Gedas Bertasius <sup>1</sup> Heng Wang <sup>1</sup> Lorenzo Torresani <sup>12</sup>" (Title)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary
The paper presents "a convolution-free approach to video classification built exclusively on self-attention over space and time" by adapting a Transformer-style model to video clips ("Abstract").

## 3. Tasks Evaluated

### Task 1: Action recognition / video classification
Task name: Action recognition / video classification. Task type: Classification. Dataset(s) used: Kinetics-400; Kinetics-600; Something-Something-V2; Diving-48. Domain: Video (human action videos). Evidence: "We evaluate TimeSformer on four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-Something-V2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (Section 4. Experiments). "For all of these datasets, we use standard classification accuracy as our main performance metric." (Appendix: Datasets). "Kinetics-400 (Carreira & Zisserman, 2017) consists of 240K training videos and 20K validation videos that span 400 human action categories." (Appendix: Datasets).

### Task 2: Long-term task classification (instructional videos)
Task name: Long-term task classification / long-term video modeling. Task type: Classification. Dataset(s) used: HowTo100M (subset with >=100 examples, 1059 task categories). Domain: Video (instructional web videos). Evidence: "Lastly, we evaluate TimeSformer on the task of long-term video modeling using HowTo100M (Miech et al., 2019)." (Section 4.6. Long-Term Video Modeling). "HowTo100M is an instructional video dataset that contains around 1M instructional Web videos showing humans performing over 23K different tasks, such as cooking, repairing, making arts, etc." (Section 4.6. Long-Term Video Modeling). "This gives a subset of HowTo100M corresponding to 120K videos spanning 1059 task categories." (Section 4.6. Long-Term Video Modeling). "Long-term task classification on HowTo100M. Given a video spanning several minutes, the goal is to predict the long-term task demonstrated in the video (e.g., cooking breakfast, cleaning house, etc)." (Table 8 caption).

## 4. Domain and Modality Scope
- Single domain? No; evaluation spans multiple video datasets, e.g., "four popular action recognition datasets: Kinetics-400 (Carreira & Zisserman, 2017), Kinetics-600 (Carreira et al., 2018), Something-Something-V2 (Goyal et al., 2017b), and Diving-48 (Li et al., 2018)." (Section 4. Experiments) and "HowTo100M is an instructional video dataset that contains around 1M instructional Web videos showing humans performing over 23K different tasks, such as cooking, repairing, making arts, etc." (Section 4.6. Long-Term Video Modeling).
- Multiple domains within the same modality? Yes; action recognition datasets and instructional-task videos are both video, as shown above.
- Multiple modalities? No; input is RGB video only: "The TimeSformer takes as input a clip  $X \in \mathbb{R}^{H \times W \times 3 \times F}$  consisting of F RGB frames of size  $H \times W$  sampled from the original video." (Section 3. The TimeSformer Model).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Action recognition / video classification (Kinetics-400/600, Something-Something-V2, Diving-48) | Not specified. | Yes; ImageNet pretraining then training on video data. | Yes; classification MLP head described. | "We adopt the \"Base\" ViT architecture (Dosovitskiy et al., 2020) pretrained on either ImageNet-1K or ImageNet-21K, as specified for each experiment." (Section 4. Experiments). "Due to a large number of parameters, training our model from scratch is difficult. Thus, before training TimeSformer on video data, we initialize it with weights learned from ImageNet." (Section 4.2. Comparison to 3D CNNs). "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (Section 3. The TimeSformer Model). |
| Long-term task classification (HowTo100M) | Not specified. | Yes; pretrained on Kinetics-400 then finetuned on HowTo100M. | Yes; classification MLP head described. | "All models in this comparison are pretrained on Kinetics-400 before finetuning on HowTo100M." (Section 4.6. Long-Term Video Modeling). "On top of this representation we append a 1-hidden-layer MLP, which is used to predict the final video classes." (Section 3. The TimeSformer Model). |

## 6. Input and Representation Constraints
- Input modality and dimensionality: "The TimeSformer takes as input a clip  $X \in \mathbb{R}^{H \times W \times 3 \times F}$  consisting of F RGB frames of size  $H \times W$  sampled from the original video." (Section 3. The TimeSformer Model).
- Patch-based representation with fixed patching: "we decompose each frame into N non-overlapping patches, each of size  $P \times P$ , such that the N patches span the entire frame, i.e.,  $N = HW/P^2$." (Section 3. The TimeSformer Model). "The patch size is  $16 \times 16$  pixels." (Section 4. Experiments). "We also experimented with a different patch size, i.e., P=32." (Section 4.7. Additional Ablations).
- Default and variant clip sizes: "Unless differently indicated, we use clips of size  $8 \times 224 \times 224$ , with frames sampled at a rate of 1/32." (Section 4. Experiments). "TimeSformer-HR, a high spatial resolution variant that operates on  $16\times448\times448$  video clips, and lastly (3) **TimeSformer-L**, a long-range configuration of our model that operates on  $96\times224\times224$  video clips with frames sampled at a rate of 1/4." (Section 4.2. Comparison to 3D CNNs).
- Token count varies with resolution and frames: "Increasing the spatial resolution results in a higher number of patches (N) per frame. The number of input tokens is also increased when using more frames." (Section 4.3. Varying the Number of Tokens).
- Resizing/cropping requirements (padding not specified): "During training, we first resize the shorter side of the video to a random value in [256, 320]. We then randomly sample a  $224 \times 224$  crop from the resized video." (Appendix A. Implementation Details). "During inference, we sample a single temporal clip in the middle of the video. We scale the shorter spatial side of a video to 224 pixels (or 448 for TimeSformer-HR) and take 3 crops of size  $224 \times 224$  ( $448 \times 448$  for TimeSformer-HR) to cover a larger spatial extent within the clip." (Appendix A. Implementation Details). Padding not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length tested (frames): "Due to GPU memory constraints, we are not able to test our model on clips longer than 96 frames." (Section 4.3. Varying the Number of Tokens).
- Sequence length is variable with input settings: "Increasing the spatial resolution results in a higher number of patches (N) per frame. The number of input tokens is also increased when using more frames." (Section 4.3. Varying the Number of Tokens).
- Global (joint) attention is considered: "One downside of self-attention in standard Transformer is that it requires computing a similarity measure for all pairs of tokens." (Section 3. The TimeSformer Model).
- Cost-reduction mechanisms include space-only and divided attention: "We can reduce the computational cost by replacing the spatiotemporal attention of Eq. 5 with spatial attention within each frame only" (Section 3. The TimeSformer Model). "We propose a more efficient architecture for spatiotemporal attention, named \"Divided Space-Time Attention\" (denoted with T+S), where temporal attention and spatial attention are separately applied one after the other." (Section 3. The TimeSformer Model). "Divided Attention performs only (N+F+2) comparisons per patch." (Section 3. The TimeSformer Model).
- Additional scalable attention structures: "(L+G) first computes a local attention ... and then calculates a sparse global attention over the entire clip using a stride of 2 patches along the temporal dimension and also the two spatial dimensions." (Section 3. The TimeSformer Model). "Finally, \"Axial\" attention decomposes the attention computation in three distinct steps: over time, width and height." (Section 3. The TimeSformer Model).

## 8. Positional Encoding (Critical Section)
- Mechanism: " $\mathbf{e}_{(p,t)}^{pos} \in \mathbb{R}^D$  represents a learnable positional embedding added to encode the spatiotemporal position of each patch." (Section 3. The TimeSformer Model).
- Where applied: "$$\mathbf{z}_{(p,t)}^{(0)} = E\mathbf{x}_{(p,t)} + \mathbf{e}_{(p,t)}^{pos}$$" (Eq. 1, Section 3. The TimeSformer Model) indicates addition at the input embedding stage.
- Variants compared: "we also conduct experiments with a few variants of TimeSformer that use: (1) no positional embedding, (2) space-only positional embedding, and (3) space-time positional embedding." (Section 4.4. The Importance of Positional Embeddings).
- Reported best variant: "The version of TimeS-former using space-time positional embeddings yields the highest accuracy on both Kinetics-400 and SSv2." (Table 4 caption).

## 9. Positional Encoding as a Variable
Positional encoding is treated as an experimental variable rather than a fixed assumption: "we also conduct experiments with a few variants of TimeSformer that use: (1) no positional embedding, (2) space-only positional embedding, and (3) space-time positional embedding." (Section 4.4. The Importance of Positional Embeddings). Multiple positional encodings are explicitly compared in "Table 4. Ablation on positional embeddings." (Table 4 caption). The paper does not state that positional encoding is "not critical" or secondary; such a claim is not stated.

## 10. Evidence of Constraint Masking
- Model size emphasis: "TimeSformer has a large learning capacity (the number of parameters is 121.4M)." (Section 4.2. Comparison to 3D CNNs).
- Reliance on pretraining for scale: "Due to a large number of parameters, training our model from scratch is difficult. Thus, before training TimeSformer on video data, we initialize it with weights learned from ImageNet." (Section 4.2. Comparison to 3D CNNs).
- Dataset sizes are large and explicitly quantified: "Kinetics-400 ... consists of 240K training videos and 20K validation videos"; "Kinetics-600 ... has 392K training videos and 30K validation videos"; "Something-Something-V2 ... contains 170K training videos and 25K validation videos"; "Diving-48 ... has 16K training videos and 3K testing videos" (Appendix: Datasets).
- Data-scale experiments: "we trained TimeSformer on different subsets of K400 and SSv2: {25%, 50%, 75%, 100%} of the full datasets." (Section 4.2. Comparison to 3D CNNs).
- Performance gains with more tokens: "increasing the spatial resolution (up to a certain point) leads to a boost in performance. Similarly, we observe that increasing the length of the input clip leads to consistent accuracy gains." (Section 4.3. Varying the Number of Tokens).

## 11. Architectural Workarounds
- Patchified inputs to manage tokenization: "we decompose each frame into N non-overlapping patches, each of size  $P \times P$ , such that the N patches span the entire frame." (Section 3. The TimeSformer Model).
- Space-only attention to reduce cost: "We can reduce the computational cost by replacing the spatiotemporal attention of Eq. 5 with spatial attention within each frame only." (Section 3. The TimeSformer Model).
- Divided Space-Time Attention to cut comparisons: "Divided Attention performs only (N+F+2) comparisons per patch." (Section 3. The TimeSformer Model).
- Sparse Local Global attention: "(L+G) first computes a local attention ... and then calculates a sparse global attention over the entire clip using a stride of 2 patches along the temporal dimension and also the two spatial dimensions." (Section 3. The TimeSformer Model).
- Axial attention decomposition: "\"Axial\" attention decomposes the attention computation in three distinct steps: over time, width and height." (Section 3. The TimeSformer Model).

## 12. Explicit Limitations and Non-Claims
- "Due to GPU memory constraints, we are not able to test our model on clips longer than 96 frames." (Section 4.3. Varying the Number of Tokens).
- "Given that our \"Base\" model already has 121M parameters, we suspect that the current datasets are not big enough to justify a further increase in model capacity." (Section 4.7. Additional Ablations).
- "We did not train any models with P values lower than 16 as those models have a much higher computational cost." (Section 4.7. Additional Ablations).
- "In the future, we plan to extend our method to other video analysis tasks such as action localization, video captioning and question-answering." (Section 5. Conclusion).

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: multiple video datasets (action recognition and instructional videos), single modality (RGB video).
> - Task structure: classification of actions and long-term tasks, no multimodal objectives described.
> - Representation rigidity: fixed patch-based tokenization (P=16 default), fixed clip sizes per experiment with explicit resizing/cropping.
> - Model sharing vs specialization: pretrained on ImageNet or Kinetics-400, then trained per dataset; no joint multi-task training described.
> - Role of positional encoding: learned spatiotemporal embeddings added at input and explicitly ablated vs none/space-only.

### 14. Final Classification
**Multi-task, single-domain**. The paper evaluates multiple classification tasks across several video datasets, e.g., "four popular action recognition datasets" (Section 4. Experiments) and "the task of long-term video modeling using HowTo100M" (Section 4.6. Long-Term Video Modeling). All evaluations are within the video modality with RGB frame inputs, and no cross-modal or open-world multi-domain claims are made.
