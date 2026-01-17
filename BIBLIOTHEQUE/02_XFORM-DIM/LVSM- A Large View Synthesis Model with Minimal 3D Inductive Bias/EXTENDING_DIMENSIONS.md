## 1. Basic Metadata

- Title: LVSM: A LARGE VIEW SYNTHESIS MODEL WITH MINIMAL 3D INDUCTIVE BIAS
- Authors: Haian Jin; Hanwen Jiang; Hao Tan; Kai Zhang; Sai Bi; Tianyuan Zhang; Fujun Luan; Noah Snavely; Zexiang Xu
- Year: Year not specified.
- Venue: Venue not specified.


## 2. One-Sentence Contribution Summary

LVSM introduces transformer-based architectures for novel view synthesis from sparse posed images without predefined 3D representations, targeting scalable and generalizable view synthesis.


## 3. Tasks Evaluated

- Task name: Object-level novel view synthesis
- Task type: Generation; Reconstruction
- Dataset(s) used: Objaverse (train), Google Scanned Objects (GSO) (test), Amazon Berkeley Objects (ABO) (test)
- Domain: Object-level images of 3D objects (rendered views)
- Quotes:
  - "We propose the Large View Synthesis Model (LVSM), a novel transformer-based approach for scalable and generalizable novel view synthesis from sparse-view inputs." (Abstract)
  - "**Object-level Datasets.** We use the Objaverse dataset (Deitke et al., 2023) to train LVSM. We follow the rendering settings in GS-LRM (Zhang et al., 2024) and render 32 random views of 730K objects. We test on two object-level datasets, Google Scanned Objects (Downs et al., 2022) (GSO) and Amazon Berkeley Objects (Collins et al., 2022b) (ABO)." (Sec. 4.1 Datasets)
  - "Following Instant3D (Li et al., 2023) and GS-LRM (Zhang et al., 2024), we use 4 sparse views as test inputs and another 10 views as target images." (Sec. 4.1 Datasets)

- Task name: Scene-level novel view synthesis
- Task type: Generation; Reconstruction
- Dataset(s) used: RealEstate10K
- Domain: Scene-level images (indoor and outdoor scenes)
- Quotes:
  - "We train (and evaluate) LVSM on object-level and scene-level datasets separately." (Sec. 4.1 Datasets)
  - "**Scene-level Datasets.** We use the RealEstate10K dataset (Zhou et al., 2018), which contains 80K video clips curated from 10K Youtube videos of both indoor and outdoor scenes." (Sec. 4.1 Datasets)
  - "For scene-level experiments We use 2 input views and 6 target views for each training example." (Appendix A.2)


## 4. Domain and Modality Scope

- Single domain? No; evaluation spans multiple domains within the same modality. Evidence: "We train (and evaluate) LVSM on object-level and scene-level datasets separately." (Sec. 4.1 Datasets)
- Multiple modalities? No; inputs are RGB images with camera poses/intrinsics. Evidence: "Given N sparse input images with known camera poses and intrinsics, denoted as  $\{(\mathbf{I}_i, \mathbf{E}_i, \mathbf{K}_i) | i = 1, \dots, N\}$ , LVSM synthesizes target image  $\mathbf{I}^t$  with novel target camera extrinsics  $\mathbf{E}^t$  and intrinsics  $\mathbf{K}^t$ . Each input image has shape  $\mathbb{R}^{H \times W \times 3}$ , where H and W are the image height and width (and there are 3 color channels)." (Sec. 3.1 Overview)
- Domain generalization or cross-domain transfer: Not claimed. The only explicit generalization is to number of input views: "our models, trained on 2 or 4 input views, demonstrate strong zero-shot generalization to an unseen number of views, ranging from a single input to more than 10." (Introduction)


## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Object-level novel view synthesis | No (trained separately). | Yes (256 -> 512 resolution fine-tuning). | Not specified. | "We train (and evaluate) LVSM on object-level and scene-level datasets separately." (Sec. 4.1 Datasets); "For object-level experiments, we use 4 input views and 8 target views for each training example by default. We first train with a resolution of 256, which takes 4 days for the *encoder-decoder* model and 7 days for the *decoder-only* model. Then, we finetune the model with a resolution of 512 for 10k iterations with a smaller learning rate of 4e-5 and a smaller total batch size of 128, which takes 2.5 days." (Appendix A.2) |
| Scene-level novel view synthesis | No (trained separately). | Yes (256 -> 512 resolution fine-tuning). | Not specified. | "We train (and evaluate) LVSM on object-level and scene-level datasets separately." (Sec. 4.1 Datasets); "For scene-level experiments We use 2 input views and 6 target views for each training example. We first train with a resolution of 256, which takes about 3 days for both *encoder-decoder* and *decoder-only* models. Then, we finetune the model with a resolution of 512 for 20k iterations with a smaller learning rate of 1e-4 and a total batch size of 128 for 3 days." (Appendix A.2) |


## 6. Input and Representation Constraints

- Input images are RGB with known camera poses/intrinsics: "Given N sparse input images with known camera poses and intrinsics, denoted as  $\{(\mathbf{I}_i, \mathbf{E}_i, \mathbf{K}_i) | i = 1, \dots, N\}$ , LVSM synthesizes target image  $\mathbf{I}^t$  with novel target camera extrinsics  $\mathbf{E}^t$  and intrinsics  $\mathbf{K}^t$ . Each input image has shape  $\mathbb{R}^{H \times W \times 3}$ , where H and W are the image height and width (and there are 3 color channels)." (Sec. 3.1 Overview)
- Inputs are patchified into non-overlapping patches with Plücker ray embeddings: "We patchify the RGB images and Plücker ray embeddings into non-overlapping patches, following the image tokenization layer of ViT (Dosovitskiy et al., 2020). We denote the image and Plücker ray patches of input image  $\mathbf{I}_i$  as  $\{\mathbf{I}_{ij} \in \mathbb{R}^{p \times p \times 3} | j=1,\dots,HW/p^2\}$  and  $\{\mathbf{P}_{ij} \in \mathbb{R}^{p \times p \times 6} | j=1,\dots,HW/p^2\}$ , respectively, where p is the patch size. For each patch, we concatenate its image patch and Plücker ray embedding patch, reshape them into a 1D vector, and use a linear layer to map it into an input patch token  $\mathbf{x}_{ij}$ :" (Sec. 3.1 Overview)
- Sequence length is defined by resolution and number of views: "We flatten the input tokens into a 1D token sequence, denoted as  $x_1,\ldots,x_{l_x}$ , where  $l_x=NHW/p^2$  is the sequence length of the input image tokens. We also flatten the target query tokens as  $q_1,\ldots,q_{l_q}$  from the ray embeddings, with  $l_q=HW/p^2$  as the sequence length." (Sec. 3.1 Overview)
- Fixed patch size and token dimension in experiments: "LVSM uses a image patch size of p=8 and token dimension d=768." (Appendix A.2)
- Fixed number of latent tokens (encoder-decoder): "The encoder-decoder LVSM has 12 encoder layers and 12 decoder layers, with 3072 latent tokens." (Appendix A.2)
- Aspect ratio / resolution generalization constraint: "Additionally, our model's performance degrades when provided with images with aspect ratios and resolutions different from those seen during training." (Appendix A.7 Limitations)
- Padding/resizing requirements: Not specified.


## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; sequence length is defined as " $l_x=NHW/p^2$ " and " $l_q=HW/p^2$ ." (Sec. 3.1 Overview)
- Fixed or variable sequence length: Input/target lengths vary with N, H, and W, while the encoder-decoder uses a fixed-length latent set: "Our encoder employs self-attention to progressively compress the information from posed input images into a fixed-length set of 1D latent tokens." (Sec. 4.4 Ablation Studies)
- Attention type: Dense, full, bidirectional self-attention with no special masks. Evidence: "we adopt dense **full self-attention** across all our encoder and decoder architectures." (Sec. 3.2); "Notably, we apply self-attention to all tokens in every transformer block of both models without introducing special attention masks or other architectural biases" (Appendix A.1)
- Computational cost mechanisms: Fixed-length latent tokens for constant decoding speed, plus training efficiency techniques. Evidence: "This design ensures a consistent rendering speed, regardless of the number of input images, as shown in Fig. 6." (Sec. 4.4); "We use FlashAttention-v2 (Dao, 2023) in the xFormers (Lefaudeux et al., 2022), gradient checkpointing (Chen et al., 2016), and mixed-precision training with Bfloat16 data type to accelerate training." (Sec. 4.2 Training Details)


## 8. Positional Encoding (Critical Section)

- Mechanism used: Plücker ray embeddings as positional embeddings. Evidence: "These latent tokens are then processed by a decoder transformer, which uses target-view Plücker rays as positional embeddings to generate the target view's image tokens" (Introduction)
- Where applied: In tokenization for both input patches and target pose tokens. Evidence: "We patchify the RGB images and Plücker ray embeddings into non-overlapping patches, following the image tokenization layer of ViT (Dosovitskiy et al., 2020). We denote the image and Plücker ray patches of input image  $\mathbf{I}_i$  as  $\{\mathbf{I}_{ij} \in \mathbb{R}^{p \times p \times 3} | j=1,\dots,HW/p^2\}$  and  $\{\mathbf{P}_{ij} \in \mathbb{R}^{p \times p \times 6} | j=1,\dots,HW/p^2\}$ , respectively, where p is the patch size. For each patch, we concatenate its image patch and Plücker ray embedding patch, reshape them into a 1D vector, and use a linear layer to map it into an input patch token  $\mathbf{x}_{ij}$ :" (Sec. 3.1 Overview); "LVSM represents the target pose to be synthesized as its Plücker ray embeddings  $\mathbf{P}^t \in \mathbb{R}^{H \times W \times 6}$ , computed from the given target extrinsics  $\mathbf{E}^t$  and intrinsics  $\mathbf{K}^t$ . We use the same patchify method and another linear layer to map it to the Plücker ray tokens of the target view, denoted as:" (Sec. 3.1 Overview)
- Fixed vs modified across experiments: Not specified; no positional-encoding ablations or alternatives are described.


## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: The paper uses Plücker ray embeddings as a fixed architectural component. Evidence: "uses target-view Plücker rays as positional embeddings" (Introduction)
- Multiple positional encodings compared: Not specified.
- Claim that PE choice is not critical or secondary: Not specified.


## 10. Evidence of Constraint Masking

- Model sizes: "| Ours Encoder-Decoder (6 + 18)  | 173M     | 26.48  | 0.901           | 0.065              | 28.32         | 0.888  | 0.117              |  |" and "| Ours Decoder-Only (24 layers)  | 171M     | 27.04  | 0.910           | 0.055              | 28.89         | 0.894  | 0.108              |  |" (Sec. 4.4, Table 2)
- Dataset sizes: "render 32 random views of 730K objects" (Sec. 4.1 Datasets); "We use the RealEstate10K dataset (Zhou et al., 2018), which contains 80K video clips curated from 10K Youtube videos of both indoor and outdoor scenes." (Sec. 4.1 Datasets)
- Scaling model size: "The experiment verifies that the decoder-only LVSM shows increasing performance when using more transformer layers" (Sec. 4.4 Ablation Studies)
- Attribution to architectural choice: "These significant performance gains validate the effectiveness of our design target of removing 3D inductive bias." (Sec. 4.3 Comparison to Baselines)
- Training scale/compute: "Our final models were trained on 64 A100 GPUs for 3-7 days, depending on the data type and model architecture," (Introduction)


## 11. Architectural Workarounds

- Fixed-length latent tokens to control decoding cost: "Our encoder employs self-attention to progressively compress the information from posed input images into a fixed-length set of 1D latent tokens. This design ensures a consistent rendering speed" (Sec. 4.4)
- Patch-based tokenization to manage input size: "We patchify the RGB images and Plücker ray embeddings into non-overlapping patches" (Sec. 3.1 Overview)
- Training stability measures: "We empirically find that using QK-Norm (Henry et al., 2020) in the transformer layers stabilizes training. We also skip optimization steps with gradient norm > 5.0" (Sec. 4.2 Training Details)
- Training efficiency techniques: "We use FlashAttention-v2 (Dao, 2023) in the xFormers (Lefaudeux et al., 2022), gradient checkpointing (Chen et al., 2016), and mixed-precision training with Bfloat16 data type to accelerate training." (Sec. 4.2 Training Details)


## 12. Explicit Limitations and Non-Claims

- Limitation (unseen regions): "Our models are deterministic, and like all prior deterministic approaches (Chen et al., 2021; Wang et al., 2021a; Sajjadi et al., 2022; Wang et al., 2023a; Zhang et al., 2024), they struggle to produce high-quality results in unseen regions." (Appendix A.7 Limitations)
- Limitation (aspect ratio / resolution shift): "our model's performance degrades when provided with images with aspect ratios and resolutions different from those seen during training." (Appendix A.7 Limitations)
- Non-claim (not generative): "Note that our LVSM models are deterministic, and thus are fundamentally different from these generative models." (Related Work)
- Future work direction: "Incorporating generative techniques or combining generative methods with our model could help solve this issue, which we leave as a promising future direction." (Appendix A.7 Limitations)


### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Object-level and scene-level NVS on RGB images; multiple domains within one modality.
> - Task structure: Novel view synthesis from sparse posed images; evaluated on object and scene datasets.
> - Representation rigidity: Patchified tokens with fixed patch size (p=8) and fixed latent token count (encoder-decoder).
> - Model sharing vs specialization: Separate training per domain; no joint multi-task training reported.
> - Role of positional encoding: Plücker ray embeddings used as positional embeddings at input; no alternatives compared.


### 14. Final Classification

**Classification:** Multi-task, multi-domain (constrained).

The evaluation spans object-level and scene-level datasets ("We train (and evaluate) LVSM on object-level and scene-level datasets separately."), indicating multiple domains within the same modality. The task itself is consistently novel view synthesis from sparse-view inputs ("novel view synthesis from sparse-view inputs"), with separate training per domain rather than unrestrained multi-task learning.
