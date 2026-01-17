## 1. Basic Metadata

- Title: "# ECHOMOTION: UNIFIED HUMAN VIDEO AND MOTION GENERATION VIA DUAL-MODALITY DIFFUSION TRANSFORMER" (Title, top of document)
- Authors: "Yuxiao Yang<sup>1,2</sup> Hualian Sheng<sup>2</sup> Sijia Cai<sup>2,*</sup> Jing Lin<sup>3</sup> Jiahao Wang<sup>4</sup> Bing Deng<sup>2</sup> Junzhe Lu<sup>1</sup> Haoqian Wang<sup>1,†</sup> Jieping Ye<sup>2,†</sup>" (Author line, top of document)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper's primary contribution is that it "introduce[s] EchoMotion, a framework designed to model the joint distribution of appearance and human motion, thereby improving the quality of complex human action video generation" (Abstract).

## 3. Tasks Evaluated

- Task name: Text-to-video generation
  - Task type: Generation
  - Dataset(s) used: "we build a new benchmark that covers a wide spectrum of human motion, from daily activities to extreme athletic feats." and "we use the automatic metrics from VBench (Huang et al., 2024) and VBench-2.0 (Zheng et al., 2025), alongside human user studies to collect numerical ratings." (Section 4.2 TEXT TO VIDEO GENERATION)
  - Domain: Human-centric video; "a wide spectrum of human motion, from daily activities to extreme athletic feats." (Section 4.2 TEXT TO VIDEO GENERATION)
  - Quotes: "Text-to-video results from EchoMotion, demonstrating both strong prompt alignment and high kinematic plausibility across a diverse range of human-centric scenarios." (Section 4.1, Figure 6 caption)

- Task name: Joint video-and-motion generation from text
  - Task type: Generation
  - Dataset(s) used: Dataset not specified.
  - Domain: Video + 3D human motion (SMPL)
  - Quotes: "This paper introduces EchoMotion, a system designed to generate videos with corresponding motion sequences from an input text prompt." (Section 3) and "EchoMotion jointly generates an SMPL motion sequence (left) and video (right), demonstrating a learned joint distribution." (Section 4.2, Figure 7 caption)

- Task name: Motion-to-video generation (cross-modal completion)
  - Task type: Generation
  - Dataset(s) used: "We provide a quantitative comparison on the motion-to-video task against several leading methods on the case from VACE-Benchmark (Jiang et al., 2025)" (Appendix A.7.2 MOTION-TO-VIDEO)
  - Domain: Video generation conditioned on motion; "synthesize a high-fidelity video that precisely follows a given motion sequence (motion-to-video)." (Section 4.3 Cross-Modal Completion)
  - Quotes: "Motion-to-video training: motion sequences serve as the conditioning input for video generation." (Section 3.2) and "synthesize a high-fidelity video that precisely follows a given motion sequence (motion-to-video)." (Section 4.3)

- Task name: Video-to-motion generation (inverse kinematics / motion recovery)
  - Task type: Reconstruction
  - Dataset(s) used: "we assess the performance of our Video-to-Motion capability on 200 samples from the 3DPW test set (Von Marcard et al., 2018)." (Appendix A.7.3 VIDEO-TO-MOTION)
  - Domain: Video -> SMPL motion parameters; "recover the underlying SMPL motion from an input video (video-to-motion)." (Section 4.3 Cross-Modal Completion)
  - Quotes: "Video-to-motion training: video sequences are used to condition motion generation." (Section 3.2) and "recover the underlying SMPL motion from an input video (video-to-motion)." (Section 4.3)

- Task name: Motion synthesis quality evaluation
  - Task type: Generation
  - Dataset(s) used: "we generated a set of 50 diverse prompts using an LLM (Team et al., 2024), rendered the generated motion parameters into mesh videos, and then asked human annotators to score them on three key aspects: Pose Plausibility (PP), Prompt Following (PF), and Motion Smoothness (MS)." (Appendix A.7.4 MOTION QUALITY)
  - Domain: 3D human motion parameters / mesh videos
  - Quotes: "While EchoMotion is primarily designed for video generation, we also evaluate its motion synthesis quality." (Appendix A.7.4) and "rendered the generated motion parameters into mesh videos" (Appendix A.7.4)

## 4. Domain and Modality Scope

- Is evaluation performed on a single domain? Yes, human-centric video and motion; HuMoVe is "a large-scale dataset of approximately 80,000 high-quality, human-centric video-motion pairs" (Abstract).
- Multiple domains within the same modality? Yes; the evaluation benchmark spans multiple human activity categories: "gymnastics and athletics; fluid, expressive motions from dance; reactive and interactive scenarios from ball and combat sports; and the natural gestures of everyday life." (Section 4.2 TEXT TO VIDEO GENERATION)
- Multiple modalities? Yes; the method "natively models the joint distribution of video and human motion modality" and uses a "text prompt" (Section 1 Introduction; Section 3).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Text-to-video generation | Yes | No per-task fine-tuning stated; trained jointly in Phase 2 after motion-only pretraining | No explicit task-specific head; task embedding added to latents | "Phase 2: Motion-video Multi-task Training. Subsequently, the model is trained on motion-video paired datasets with both branches unfrozen and active, enabling the generation of both visual and motion sequences." (Section 3.2); "a lightweight MLP projects the task embedding to the latent space. This task hint is then added to the latents to guide conditional token prediction." (Section 3.2) |
| Joint video-and-motion generation | Yes | No per-task fine-tuning stated; trained jointly in Phase 2 after motion-only pretraining | No explicit task-specific head; task embedding added to latents | "each paradigm is randomly sampled: 1) Joint training: generate both video and motion sequences concurrently." (Section 3.2); "a lightweight MLP projects the task embedding to the latent space. This task hint is then added to the latents to guide conditional token prediction." (Section 3.2) |
| Motion-to-video generation | Yes | No per-task fine-tuning stated; trained jointly in Phase 2 after motion-only pretraining | No explicit task-specific head; task embedding added to latents | "2) Motion-to-video training: motion sequences serve as the conditioning input for video generation." (Section 3.2); "Phase 2: Motion-video Multi-task Training." (Section 3.2) |
| Video-to-motion generation | Yes | No per-task fine-tuning stated; trained jointly in Phase 2 after motion-only pretraining | No explicit task-specific head; task embedding added to latents | "3) Video-to-motion training: video sequences are used to condition motion generation." (Section 3.2); "Phase 2: Motion-video Multi-task Training." (Section 3.2) |
| Motion synthesis quality (motion generation) | Yes (motion branch) | Motion-only pretraining precedes joint training | No explicit task-specific head; task embedding added to latents | "Phase 1: Motion-only Pretraining. The motion branch is trained independently using motion-only datasets, while the video branch is frozen and deactivated (inputs omitted)." (Section 3.2); "a lightweight MLP projects the task embedding to the latent space." (Section 3.2) |

## 6. Input and Representation Constraints

- Fixed/variable input resolution: Experiment settings list fixed video sizes per model, e.g., "Video height 480", "Video width 832", "Video frame 81", "FPS 16" (Appendix A.9, Table 7) and "Video height 708", "Video width 1280", "Video frame 121", "FPS 24" (Appendix A.9, Table 8).
- Fixed patch size: Not specified; only that DiT uses "An input encoder that \"patchifies\" the noisy latent variable into a sequence of tokens." (Appendix A.2)
- Fixed number of tokens: Motion tokens are explicitly fixed per frame: "generating 51 motion tokens per frame." (Section 3.1)
- Fixed dimensionality: SMPL parameters are explicitly sized: "shape parameters  $\beta \in \mathbb{R}^{10}$ ... pose parameters  $\theta \in \mathbb{R}^{24 \times 6}$ ... global body orientation  $\gamma \in \mathbb{R}^6$ , and the human root joint position  $v \in \mathbb{R}^3$ ." and "the 3D joint position  $n \in \mathbb{R}^{24 \times 3}$  to represent each human joint." (Section 3.1)
- Token concatenation assumption: "These motion tokens are concatenated with visual tokens to form a **unified multi-modal context sequence**." (Section 3.1)
- Temporal alignment/stride constraint: "recognizing the direct temporal correspondence between motion tokens and visual tokens (4x temporal compression from video VAE), we assign a scaled indexing scheme." (Section 3.1)
- Padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; only token counts are given for one setting, e.g., "4,131 motion tokens vs. 32,760 video tokens in the 1.3B model." (Appendix A.10)
- Fixed or variable sequence length: Not explicitly stated; experiment settings specify fixed frame counts such as "Video frame 81" (Appendix A.9, Table 7) and "Video frame 121" (Appendix A.9, Table 8).
- Attention type: Global joint self-attention over the concatenated sequence; "Thereafter, a joint self-attention layer is applied to capture dependencies and correlations across both modalities." (Section 3.1)
- Mechanisms to manage computational cost: temporal compression and token economy are used, including "4x temporal compression from video VAE" (Section 3.1), "typical time-level down-sampling applied to visual tokens" (Section 3.1), and the fact that motion tokens are a "small fraction of the total sequence length (e.g., 4,131 motion tokens vs. 32,760 video tokens in the 1.3B model)." (Appendix A.10). A compute-conscious hybrid design is also described: "we adopt a hybrid strategy, replacing half of the video blocks with our dual-modality blocks" for the 5B model (Appendix A.9).

## 8. Positional Encoding (Critical Section)

- Mechanism: The model uses RoPE via MVS-RoPE; "we propose MVS-RoPE (Motion-Video Syncronized RoPE), which offers unified 3D positional encoding for both video and motion tokens." (Abstract). The encoding is explicitly rotary: "The function  $\mathcal{R}(\cdot)$  represents the RoPE encoding, which applies a rotation based on the input temporal and spatial indices." (Section 3.1). Appendix A.2 further states, "Rotary Position Embedding(Heo et al., 2024) encodes absolute positional information by applying position-dependent rotations to the query and key vectors." (Appendix A.2)
- Where applied: Positional encoding is injected within attention processing; "Within self-attention layers, we propose a specialized multi-modal positional embedding to inject features with precise position information." (Section 3.1)
- Fixed across experiments or modified per task: Not explicitly stated; the paper presents MVS-RoPE as the default and evaluates alternatives (see Section 9).

## 9. Positional Encoding as a Variable

- Core variable or fixed assumption? Treated as a core architectural variable; "MVS-RoPE Design. To validate our MVS-RoPE design, we visualize the self-attention score" (Section 4.4).
- Multiple positional encodings compared? Yes; comparisons include removal and collision variants: "the baseline without MVS-RoPE (b) fails completely" (Section 4.4) and "We compare two distinct configurations: (a) EchoMotion (Proposed)... (b) Positional Collision (Baseline)." (Appendix A.8.1)
- PE claimed not critical/secondary? Not claimed.

## 10. Evidence of Constraint Masking

- Model size(s): "We perform experiments on two variants of the open-sourced base model, Wan2.1-1.3B and Wan2.2-5B" (Section 4.1). The final parameter counts are reported: "This results in a final model with 2.6B parameters" and "yields a final model with 7.5B parameters." (Appendix A.9)
- Dataset size(s): "HuMoVe, a large-scale dataset of approximately 80,000 high-quality, human-centric video-motion pairs." (Abstract) and "HumanML3D dataset ... containing over 14,616 human motions in SMPL format" (Section 4.1). Evaluation includes "200 samples from the 3DPW test set" (Appendix A.7.3).
- Attribution of gains: The paper argues gains are not just from scale: "the lack of anatomical plausibility is not merely a matter of data scale" (Section 1 Introduction). It explicitly attributes improvements to joint modeling: "The key to high-quality human motion synthesis lies in the joint modeling of appearance and kinematics during training, whereas simply adding more human-centric video data offers marginal benefits." (Section 4.4)
- Scale/data/compute limitations vs. closed models: For commercial comparisons, the gap is attributed to scale and data: "a gap exists compared to top-tier closed-source models... vast disparities in model scale, training data, and computational resources." (Appendix A.5)
- Training tricks: The method uses staged training: "a two-stage training recipe: an initial motion-only training phase followed by a motion-video multi-task training phase." (Section 1 Introduction)

## 11. Architectural Workarounds

- Dual-branch architecture for modality separation: "a dual-branch architecture that jointly processes tokens concatenated from different modalities." (Abstract) This is implemented with separate projections: "modality-specific projections, implemented as two distinct sets of learnable matrices." (Section 3.1)
- MVS-RoPE for alignment and collision avoidance: "MVS-RoPE (Motion-Video Syncronized RoPE), which offers unified 3D positional encoding for both video and motion tokens" and provides "a synchronized coordinate system for the dual-modal latent sequence" that "fosters temporal alignment between the two modalities." (Abstract)
- Two-stage training strategy to stabilize multimodal learning: "an initial motion-only training phase followed by a motion-video multi-task training phase" (Section 1 Introduction) and "Phase 1: Motion-only Pretraining... Phase 2: Motion-video Multi-task Training." (Section 3.2)
- Task conditioning via embeddings rather than new heads: "a lightweight MLP projects the task embedding to the latent space. This task hint is then added to the latents to guide conditional token prediction." (Section 3.2)
- Hybrid block replacement to limit compute: "we adopt a hybrid strategy, replacing half of the video blocks with our dual-modality blocks" for the 5B model. (Appendix A.9)
- Token/temporal efficiency choices: The motion representation is "more token-efficient" (Section 1 Introduction), and the method relies on "4x temporal compression from video VAE" plus "time-level down-sampling applied to visual tokens" (Section 3.1).

## 12. Explicit Limitations and Non-Claims

- Limitation: "Our framework is currently limited to single-person generation." (Section 5 Conclusion, Limitations)
- Future work / non-claim: "Extending it to multiperson scenarios... leave multi-person generation as a promising direction for future projects." (Section 5 Conclusion, Limitations)
- Other explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Human-centric video-motion data, "human-centric video-motion pairs" with diverse action categories (Abstract; Section 4.2).
> - Task structure: Multi-task cross-modal generation and completion, "Joint training... Motion-to-video... Video-to-motion" (Section 3.2).
> - Representation rigidity: Fixed SMPL parameter dimensions ("$\beta \in \mathbb{R}^{10}$" etc.) and fixed experiment video sizes ("Video height 480"; "Video frame 81"). (Section 3.1; Appendix A.9)
> - Model sharing vs specialization: Single model trained jointly in Phase 2 ("model is trained on motion-video paired datasets with both branches unfrozen and active"). (Section 3.2)
> - Role of positional encoding: Central architectural component, "MVS-RoPE... unified 3D positional encoding" with ablations. (Abstract; Section 4.4)

### 14. Final Classification

**Multi-task, single-domain.** The experiments span multiple tasks within one human-centric domain, including "Joint training... Motion-to-video... Video-to-motion" (Section 3.2) and evaluations on "human-centric video-motion pairs" (Abstract). The setup is multi-modal but not multi-domain; no domain generalization or cross-domain transfer is claimed.
