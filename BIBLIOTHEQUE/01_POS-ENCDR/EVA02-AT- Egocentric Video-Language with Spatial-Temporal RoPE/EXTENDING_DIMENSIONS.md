## 1. Basic Metadata

- Title: "EVA02-AT: Egocentric Video-Language Understanding with Spatial-Temporal Rotary Positional Embeddings and Symmetric Optimization" (Title line)
- Authors: "Xiaoqi Wang, Student Member, IEEE, Yi Wang, Member, IEEE, Lap-Pui Chau, Fellow, IEEE" (Author line)
- Year: Year not specified.
- Venue: Venue not specified.

---

## 2. One-Sentence Contribution Summary

Introduction: "To address these issues, we propose EVA-02 with spAtial-**Temporal attention** (EVA02-AT), a training-efficient solution for egocentric video understanding tasks."

---

## 3. Tasks Evaluated

- Task name: EgoMCQ (Ego4D Multiple-Choice Questions)
  - Task type: Classification
  - Dataset(s): Ego4D (EgoMCQ benchmark)
  - Domain: Egocentric video-language
  - Quotes: "After pretraining, we evaluate models on the Ego4D Multiple-Choice Questions (EgoMCQ) benchmark." (Section V.A Datasets and Implementation Details)

- Task name: EK-100 MIR (multi-instance retrieval)
  - Task type: Other (specify: video-text retrieval / multi-instance retrieval)
  - Dataset(s): Epic-Kitchens-100 (EK-100)
  - Domain: Egocentric video-language
  - Quotes: "Before fine-tuning, we directly evaluate the pretrained model on EK-100's multi-instance retrieval (MIR) challenge and the Charades-Ego action recognition challenge, where the performance will be treated as zero-shot results." (Section V.A Datasets and Implementation Details)

- Task name: Charades-Ego action recognition (video-to-text action recognition)
  - Task type: Classification
  - Dataset(s): Charades-Ego
  - Domain: Egocentric video-language
  - Quotes: "Before fine-tuning, we directly evaluate the pretrained model on EK-100's multi-instance retrieval (MIR) challenge and the Charades-Ego action recognition challenge, where the performance will be treated as zero-shot results." (Section V.A Datasets and Implementation Details); "CharadesEgo Action Recognition. Table II provides the comparison results on CharadesEgo Video-to-Text action recognition task." (Section V.B Compare with State-of-the-Arts)

---

## 4. Domain and Modality Scope

- Domain scope: Single domain (egocentric video) across datasets. Evidence: "We conduct the experiments on three egocentric datasets: Ego4D, Epic-Kitchens-100 (EK-100), and Charades-Ego." (Section V.A Datasets and Implementation Details)
- Modality scope: Multiple modalities (video and text). Evidence: "EVA02-AT: Egocentric Video-Language Understanding with Spatial-Temporal Rotary Positional Embeddings and Symmetric Optimization" (Title line); "the objective of the video text retrieval task is to learn a similarity calculation function  $S(\cdot)$  that satisfies  $S(\mathcal{V}, \mathcal{T}) = C$ ." (Section III Preliminary, Learning objective)
- Domain generalization or cross-domain transfer: The paper claims "achieves the generalized egocentric video representations" (Section VI Conclusion). Cross-domain transfer beyond egocentric video is not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| EgoMCQ (Ego4D Multiple-Choice Questions) | Yes (pretrained EVA02-AT weights) | No (direct evaluation) | Not specified. | "After pretraining, we evaluate models on the Ego4D Multiple-Choice Questions (EgoMCQ) benchmark." (Section V.A Datasets and Implementation Details) |
| EK-100 MIR | Yes (pretrained then fine-tuned) | Yes | Not specified. | "Before fine-tuning, we directly evaluate the pretrained model on EK-100's multi-instance retrieval (MIR) challenge and the Charades-Ego action recognition challenge, where the performance will be treated as zero-shot results. After that, we fine-tune the pretrained model on the training set of these two benchmarks, respectively, and evaluate their fine-tuned results." (Section V.A Datasets and Implementation Details) |
| Charades-Ego action recognition | Yes (pretrained then fine-tuned) | Yes | Not specified. | "Before fine-tuning, we directly evaluate the pretrained model on EK-100's multi-instance retrieval (MIR) challenge and the Charades-Ego action recognition challenge, where the performance will be treated as zero-shot results. After that, we fine-tune the pretrained model on the training set of these two benchmarks, respectively, and evaluate their fine-tuned results." (Section V.A Datasets and Implementation Details) |

---

## 6. Input and Representation Constraints

- Input shape and tokenization: "For patch embedding, an input video sequence  $\mathbf{v} \in \mathbb{R}^{C \times T \times H \times W}$ , where C, T, H, W represents channels, number of frames, height, and length, is processed in the spatial domain only. This approach ensures compatibility with the original image encoder, yielding a patchified feature of dimension  $\mathbb{R}^{B \times (T \times P^2) \times D}$ , where  $D = \frac{CHW}{p^2}$ ." (Section IV.A EVA-02 AT Transformer, Patchify)
- Learnable positional embeddings tied to T and p^2: "We introduce two distinct learnable positional embeddings: a temporal positional embedding  $P_t \in \mathbb{R}^{T \times D}$  and a spatial positional embedding  $P_{xy} \in \mathbb{R}^{p^2 \times D}$ ." (Section IV.A EVA-02 AT Transformer, Patchify)
- Patch size / tube convolution: "we employ a 3D convolution, also known as tube convolution [12], with a convolution kernel of  $1 \times p \times p$ ." (Section IV.A EVA-02 AT Transformer, Patchify)
- Fixed input resolution in experiments: "During both pretraining and fine-tuning, frames are sampled uniformly from each clip at a resolution of  $3 \times 224 \times 224$ , and the dimension of the feature space is set to 256." (Section V.A Implementation Details)
- Fixed number of frames per setting: "We evenly sample 4 frames for each video clip." (Section V.A Ego4D pretraining); "During fine-tuning, 16 frames are sampled for each video clip." (Section V.A EK-100 MIR)
- Padding/resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; frame counts used in experiments are explicit. Evidence: "We evenly sample 4 frames for each video clip." (Section V.A Ego4D pretraining); "During fine-tuning, 16 frames are sampled for each video clip." (Section V.A EK-100 MIR)
- Fixed or variable length: Fixed per experiment (explicit frame counts above); no variable-length policy stated.
- Attention type: Global, joint spatial-temporal. Evidence: "joint attention blocks that process both spatial and temporal information simultaneously" (Section IV.A Joint Spatial-Temporal Attention) and "the attention score between query and key becomes a global attention among all the patches in the video clip instead of the spatial attention on a single frame." (Section IV.A Joint Spatial-Temporal Attention)
- Mechanisms to manage computational cost (windowing/pooling/pruning): Not specified.

---

## 8. Positional Encoding (Critical Section)

- Mechanism: Spatial-temporal RoPE combined with learnable temporal and spatial positional embeddings. Evidence: "To achieve this, we extend the Rotary Positional Embedding (RoPE) to a spatial-temporal approach that is compatible with the original 2D-RoPE." and "Therefore, we first generate a 1D-RoPE for the temporal embeddings and a 2D-RoPE for the spatial embeddings, where the dimension of both embeddings corresponds to the whole feature dimension. Then, we conduct an inner product of the temporal and spatial RoPEs to obtain the final representations of our spatial-temporal RoPE. This approach combines the RoPE with learnable temporal and spatial positional embeddings, forming a final positional embedding." (Section I Introduction)
- Learnable positional embeddings at input: "We introduce two distinct learnable positional embeddings: a temporal positional embedding  $P_t \in \mathbb{R}^{T \times D}$  and a spatial positional embedding  $P_{xy} \in \mathbb{R}^{p^2 \times D}$ ." (Section IV.A EVA-02 AT Transformer, Patchify)
- Application location: RoPE applied in attention (Q/K) at each joint-attention layer. Evidence: "Since we use the standard QK-RoPE, the output of our joint spatial-temporal attention at k-th layer can be expressed as: $$= Attn\left(R_{(xy+t)}W_{q}z^{k-1}, R_{(xy+t)}W_{k}z^{k-1}, W_{v}z^{k-1}\right).$$" (Section IV.A Joint Spatial-Temporal Attention)
- Fixed vs modified: Temporal positional embedding choices are compared in ablations. Evidence: "TABLE IV Comparison between different temporal embeddings on the zero-shot EK-100 MIR benchmark." (Section V.C Ablation Study)

---

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Treated as a research variable in ablations. Evidence: "To evaluate the effectiveness of both our EVA02-AT network and the SMS loss function, we conduct the ablation experiments from three aspects: (1) the zero-shot performance across different network architectures; (2) the EVA02-AT model with different temporal positional embedding choices; (3) the fine-tuned performance across different loss functions." (Section V.C Ablation Study)
- Multiple positional encodings compared: "In table IV, we change the temporal positional embedding to (a) the learnable positional embedding, (b) 1D-RoPE embedding, and (c) learnable positional embedding with RoPE embedding." (Section V.C Ablation Study)
- PE choice claimed non-critical or secondary: "changing the temporal positional embedding will not influence the performance significantly" (Section V.C Ablation Study)

---

## 10. Evidence of Constraint Masking

- Dataset scale: "the EgoClip is proposed by EgoVLP [8], which contains 3.8 million videotext pairs for training" and "The EgoClip+ is proposed by LaViLa [10], which has a 35-million corpus that is augmented by GPT-2 XL [46]." (Section V.A Datasets and Implementation Details)
- Model size evidence: "The 'Params (M)' column lists the number of parameters for video encoders, text encoders, and additional blocks (if any), in that order." (Section V.B Compare with State-of-the-Arts, Table III caption); example entry: "| EVA02-AT  | EgoClip    | CLIP-EVA02-AT-L            | 304 + 124     | 42.1              | 35.0              | 38.5        | 37.2               | 33.9              | 35.5        |" (Table III)
- Scaling model size: "Scaling to a large-size model, the gain boosts to 9.0% in average mAP (63.5 vs. 54.5), and 5.2% (74.2 vs. 69.0) in average nDCG." (Section V.B Compare with State-of-the-Arts)
- Scaling data / pretraining choice: "The choice of pretraining data critically affects performance on the EK-100 multi-instance retrieval task." (Section V.B Compare with State-of-the-Arts)
- Training tricks / loss: "We can also observe from the table that our SMS loss drives much of this improvement. Simply replacing AVION's MI-MM loss with SMS yields a 7.6% improvement in average mAP and a 4.0% improvement in average nDCG." (Section V.B Compare with State-of-the-Arts)

---

## 11. Architectural Workarounds

- Single-stage pretraining to reduce cost: "simplifying the pretraining pipeline to a single stage by directly transferring the image-based CLIP model to a video-based one through video-text alignment." (Section I Introduction)
- Joint attention blocks vs divided attention: "joint attention blocks that process both spatial and temporal information simultaneously are adopted, rather than the divided spatial and temporal attention used in typical video encoders" (Section IV.A Joint Spatial-Temporal Attention)
- Integrated spatial-temporal RoPE on full dimension: "we thus apply the spatial RoPE and temporal RoPE on the entire dimension instead of manually dividing the dimension into uneven slides." (Section IV.A Joint Spatial-Temporal Attention)
- Spatial-only patchify for compatibility: "is processed in the spatial domain only. This approach ensures compatibility with the original image encoder" (Section IV.A EVA-02 AT Transformer, Patchify)
- Tube convolution in patch embedding: "we employ a 3D convolution, also known as tube convolution [12], with a convolution kernel of  $1 \times p \times p$ ." (Section IV.A EVA-02 AT Transformer, Patchify)

---

## 12. Explicit Limitations and Non-Claims

- Limitations: Not specified.
- Non-claims: Not specified.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Egocentric video-language across Ego4D, EK-100, and Charades-Ego.
> - Task structure: Multiple supervised evaluation tasks (EgoMCQ, multi-instance retrieval, action recognition) rather than open-ended multi-domain settings.
> - Representation rigidity: Fixed input resolution (3 x 224 x 224) and fixed frame counts per experiment (4 or 16), with patchified tokens tied to T and p^2.
> - Model sharing vs specialization: Single pretrained model evaluated zero-shot and then fine-tuned separately per downstream benchmark.
> - Role of positional encoding: Central architectural component (spatial-temporal RoPE) and explicitly varied in ablations.

---

### 14. Final Classification

**Multi-task, single-domain.** The evaluation spans multiple tasks: "Ego4D Multiple-Choice Questions (EgoMCQ)" plus "EK-100's multi-instance retrieval (MIR) challenge" and "the Charades-Ego action recognition challenge" (Section V.A Datasets and Implementation Details). All evaluations are within egocentric video datasets: "three egocentric datasets: Ego4D, Epic-Kitchens-100 (EK-100), and Charades-Ego." (Section V.A Datasets and Implementation Details)
