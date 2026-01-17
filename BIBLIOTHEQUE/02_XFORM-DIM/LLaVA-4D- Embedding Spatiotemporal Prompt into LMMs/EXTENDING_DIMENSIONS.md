## 1. Basic Metadata

- Title: "LLaVA-4D: Embedding SpatioTemporal Prompt into LMMs for 4D Scene Understanding" (Title)
- Authors: "Hanyu Zhou<sup>1</sup>, Gim Hee Lee<sup>1</sup>" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes "LLaVA-4D, a general LMM framework with a novel spatiotemporal prompt for visual representation in 4D scene understanding" to address that existing 3D LMMs "fail to capture temporally varying dynamic objects" (Abstract).

---

## 3. Tasks Evaluated

Task name: Dense captioning (DC)
Task type: Generation
Dataset(s) used: Scan2Cap; Chat4D
Domain: "multiple 3D datasets" and "2D, 3D and 4D vision-language data types" (multi-view videos)
Quotes: "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." (Section 4.1 Our Chat4D Dataset) "We compare all competing methods on multiple 3D datasets: Scan2Cap [43], ScanQA [41], ScanRef [55] and Multi3DRefer [44] and our Chat4D dataset." (Section 5.1 Experiment Setup) "We evaluate the quality of generated text response for Scan2Cap and ScanQA in terms of CiDEr (C), BLEU-4 (B-4), METEOR (M)." (Section 5.1 Experiment Setup)

Task name: Visual question answering (QA)
Task type: Reasoning / relational; Generation
Dataset(s) used: ScanQA; Chat4D
Domain: "multiple 3D datasets" and "2D, 3D and 4D vision-language data types" (multi-view videos)
Quotes: "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." (Section 4.1 Our Chat4D Dataset) "We compare all competing methods on multiple 3D datasets: Scan2Cap [43], ScanQA [41], ScanRef [55] and Multi3DRefer [44] and our Chat4D dataset." (Section 5.1 Experiment Setup) "We evaluate the quality of generated text response for Scan2Cap and ScanQA in terms of CiDEr (C), BLEU-4 (B-4), METEOR (M)." (Section 5.1 Experiment Setup)

Task name: Visual grounding (VG)
Task type: Detection; Other (grounding)
Dataset(s) used: Multi3DRefer; ScanRef; Chat4D
Domain: "multiple 3D datasets" and "2D, 3D and 4D vision-language data types" (multi-view videos)
Quotes: "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." (Section 4.1 Our Chat4D Dataset) "We compare all competing methods on multiple 3D datasets: Scan2Cap [43], ScanQA [41], ScanRef [55] and Multi3DRefer [44] and our Chat4D dataset." (Section 5.1 Experiment Setup) "We choose the F1 metric of object prediction precision for Multi3DRefer, and the accuracy of intersection over unions for grounding task from ScanRef." (Section 5.1 Experiment Setup) "The metrics are also applicable to the evaluation on our Chat4D, where grounding accuracy is divided into spatial and temporal components: S/TAcc." (Section 5.1 Experiment Setup)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Multiple domains within the same modality (vision) are used, including 3D datasets and a 4D dataset. Evidence: "We compare all competing methods on multiple 3D datasets: Scan2Cap [43], ScanQA [41], ScanRef [55] and Multi3DRefer [44] and our Chat4D dataset." (Section 5.1 Experiment Setup) and "our dataset includes 2D, 3D and 4D vision-language data types" (Section 4.1 Our Chat4D Dataset).
- Modalities: Multiple modalities (vision and language). Evidence: "Large multimodal models (LMMs) [1, 2] aim to learn the representation alignment between language and other modalities such as vision [3] and audio [4]." (Section 1 Introduction) and "Additionally, we present *Chat4D*, a 4D vision-language dataset with spatiotemporal coordinate annotations designed to instruction-tune our model for more effective 4D scene understanding." (Section 1 Introduction).
- Domain generalization or cross-domain transfer: The paper mentions generalization for 4D understanding but does not explicitly claim cross-domain transfer. Evidence: "we use 4D vision-language data of Chat4D to enhance the generalization of our model for fine-grained spatiotemporal understanding with 4D coordinates through a multi-task instruction fine-tuning strategy." (Section 4.2 Training Pipeline). Cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks

Training is staged across tasks within a single model: "**Stage 1: Content Alignment.** The training sets of the DC and QA tasks in the 2D&3D vision-language data of our Chat4D are used to initially align the content between visual and linguistic representations." "**Stage 2: Spatiotemporal Coordinate Alignment.** In order to further improve the fine-grained understanding capability of our model under the spatiotemporal coordinate prompt, we use the training data of the VG task in the 2D&3D vision-language subset of our Chat4D to refine the spatiotemporal coordinate alignment between visual and linguistic representations." "**Stage 3: 4D Task Instruction Fine-Tuning.** To further improve our model for 4D scene understanding, we use 4D vision-language data of Chat4D to enhance the generalization of our model for fine-grained spatiotemporal understanding with 4D coordinates through a multi-task instruction fine-tuning strategy." (Section 4.2 Training Pipeline)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Dense captioning (DC) | Yes (single model across stages) | Yes (stage-based fine-tuning) | Not specified. | "**Stage 1: Content Alignment.** The training sets of the DC and QA tasks in the 2D&3D vision-language data of our Chat4D are used to initially align the content between visual and linguistic representations." and "**Stage 3: 4D Task Instruction Fine-Tuning.** To further improve our model for 4D scene understanding, we use 4D vision-language data of Chat4D to enhance the generalization of our model for fine-grained spatiotemporal understanding with 4D coordinates through a multi-task instruction fine-tuning strategy." (Section 4.2 Training Pipeline) |
| Visual QA | Yes (single model across stages) | Yes (stage-based fine-tuning) | Not specified. | "**Stage 1: Content Alignment.** The training sets of the DC and QA tasks in the 2D&3D vision-language data of our Chat4D are used to initially align the content between visual and linguistic representations." (Section 4.2 Training Pipeline) |
| Visual grounding (VG) | Yes (single model across stages) | Yes (stage-based fine-tuning) | Not specified. | "**Stage 2: Spatiotemporal Coordinate Alignment.** In order to further improve the fine-grained understanding capability of our model under the spatiotemporal coordinate prompt, we use the training data of the VG task in the 2D&3D vision-language subset of our Chat4D to refine the spatiotemporal coordinate alignment between visual and linguistic representations." (Section 4.2 Training Pipeline) |

---

## 6. Input and Representation Constraints

- Multi-view video input: "Given a multi-view video input sequence *I*," (Section 3 Our LLaVA-4D).
- Fixed 4D coordinate representation: "we construct 4D coordinate tensors [x,y,z,t] from multi-view videos using visual geometry, and perform spatiotemporal encoding  $PE(\cdot)$ ,  $TE(\cdot)$  on the coordinates." and "After traversing all videos, we concatenate time and corresponding 3D position to form the 4D coordinate tensor [x, y, z, t]." (Section 3 Our LLaVA-4D; Section 3.1 Dynamic-Aware 4D Coordinate Encoding)
- 3D reconstruction assumptions (SfM/MVS, camera intrinsics): "**4D Coordinate Definition.** Given an image from a certain view at timestamp t, we use SfM [29] for camera pose  $P = [R \mid T]$  and MVS [30] for depth D. Combined with intrinsic parameter K, we transform 2D pixel coordinate  $x_{2D}$  to world coordinate system via geometric projection [31, 32]:" (Section 3.1 Dynamic-Aware 4D Coordinate Encoding).
- Optical flow for temporal encoding: "The 4D coordinate encoding module constructs 4D coordinates for multi-view videos and incorporates optical flow to enhance spatiotemporal encoding." and "where vel is estimated optical flow, and  $\Phi(\cdot)$  is softmax function." (Section 3 Our LLaVA-4D; Section 3.1 Dynamic-Aware 4D Coordinate Encoding).
- Zero padding in Stage 1: "The 4D spatiotemporal coordinate features  $p_{4D}$  are temporarily set as zero padding." (Section 4.2 Training Pipeline).
- Input resolution / patch size / token count: Not specified. Model choice evidence: "Our LLaVA-4D model utilizes the pre-trained weights of LLaVA-1.5-7B [17] and the vision encoder of CLIP-ViT-L-336px [3, 59]." (Section 5.1 Experiment Setup).

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Not specified.
- Attention type: Cross-attention is used in a transformer-based fusion module. Evidence: "Cross-attention fusion module is a transformer-based network architecture [34]." (Section 5.1 Experiment Setup) and "Next, we fuse the 4D coordinate embeddings with the spatiotemporal features via a cross-attention mechanism [34, 35]:" (Section 3.2 Spatiotemporal-Disentangled Vision Embedding).
- Computational cost management (windowing/pooling/pruning): Not specified.

---

## 8. Positional Encoding (Critical Section)

- Mechanism: Learnable Fourier feature (sin/cos) encodings for spatial position and time. Evidence: "We circumvent this challenge by adopting the same spatial position encoding strategy for objects and background via learnable Fourier feature [33]:" and "have different motion patterns and thus we add motion information into the temporal encoding:" (Section 3.1 Dynamic-Aware 4D Coordinate Encoding).
- Where applied: To 4D coordinates as a spatiotemporal prompt and to textual coordinates in language tokens. Evidence: "we construct 4D coordinate tensors [x,y,z,t] from multi-view videos using visual geometry, and perform spatiotemporal encoding  $PE(\cdot)$ ,  $TE(\cdot)$  on the coordinates." (Section 3 Our LLaVA-4D) and "apply the same spatiotemporal encodings  $\mathrm{PE}(\cdot)$  and  $\mathrm{TE}(\cdot)$  to textual position tp and time tt:" (Section 3.3 Coordinate-Aligned Language Embedding).
- Fixed vs modified per task / ablated: The paper compares variants of coordinate encoding (see Section 9); no per-task positional encoding changes are explicitly described beyond these ablations.

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Treated as a core research variable with ablations. Evidence: "**Role of 4D Coordinate Encoding.** In Table 3, we analyze the impact of 3D position encoding and 1D time encoding on the performance of 4D understanding." (Section 5.3 Ablation Study and Discussion)
- Multiple positional encodings compared: Yes. Evidence (Table 3): "w/o Encoding"; "w/ 3D position<br>w/ 1D time"; "w/ 1D time"; "w/ 4D coordinate" (Section 5.3 Ablation Study and Discussion).
- Claims that PE choice is not critical: Not claimed. Instead, "Coordinate embedding is the key to improving the overall performance of 4D understanding tasks by a large margin." (Section 5.3 Ablation Study and Discussion)

---

## 10. Evidence of Constraint Masking

- Model size: "Our LLaVA-4D model utilizes the pre-trained weights of LLaVA-1.5-7B [17]" (Section 5.1 Experiment Setup).
- Dataset sizes: "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." and "produce a dataset of 224.6K samples." (Section 4.1 Our Chat4D Dataset).
- Attributed source of gains: The paper emphasizes architectural components over scale: "Coordinate embedding is the key to improving the overall performance of 4D understanding tasks by a large margin." and "Feature disentanglement improves the upper limit of 4D scene understanding to a certain extent by strengthening the representation of spatial and temporal characteristics. Feature fusion further enhances the spatiotemporal understanding ability of the LMM." (Section 5.3 Ablation Study and Discussion).
- Claims about scaling model size or data as primary driver: Not stated.

---

## 11. Architectural Workarounds

- Dynamic-aware 4D coordinate encoding with optical flow to manage spatiotemporal complexity: "The 4D coordinate encoding module constructs 4D coordinates for multi-view videos and incorporates optical flow to enhance spatiotemporal encoding." (Section 3 Our LLaVA-4D)
- Spatiotemporal-disentangled vision embedding to reduce heterogeneous feature misalignment: "2) **Spatiotemporal-Disentangled Vision Embedding** (*cf.* Sec. 3.2). This is the visual representation stage where we extract visual features f from multi-view videos using a vision encoder, and disentangle these visual features into spatiotemporal components:" and "A unified visual representation for 4D scene usually suffers from misaligned heterogeneous features, which inspires us to disentangle visual features into spatiotemporal components." (Section 3 Our LLaVA-4D; Section 3.2 Spatiotemporal-Disentangled Vision Embedding).
- Cross-attention fusion to inject 4D coordinates into features: "Next, we fuse the 4D coordinate embeddings with the spatiotemporal features via a cross-attention mechanism [34, 35]:" (Section 3.2 Spatiotemporal-Disentangled Vision Embedding).
- Multi-stage training to stabilize and align representations: "To ensure the stability of the training process and improve the performance of the model, we divide the entire training into three stages" (Section 4.2 Training Pipeline).

---

## 12. Explicit Limitations and Non-Claims

- Limitation: "While our model performs well on most 3D and 4D dynamic scenes, it struggles with fast-moving objects due to motion blur from frame-based cameras." (Section 5.3 Ablation Study and Discussion)
- Future work: "In future work, we plan to incorporate event cameras [60] with high temporal resolution to improve dynamic representation." (Section 5.3 Ablation Study and Discussion)
- Explicit non-claims about open-world, unrestrained multi-task learning, or meta-learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Evaluation uses multiple vision domains ("multiple 3D datasets" plus "2D, 3D and 4D vision-language data types"), centered on 4D scene understanding with multi-view videos.
> - Task structure: Dense captioning, visual QA, and visual grounding are the explicit tasks evaluated; 4D grounding is scored with spatial/temporal accuracy.
> - Representation rigidity: Inputs are encoded into a fixed 4D coordinate tensor [x, y, z, t] with learnable Fourier spatiotemporal encodings and optical-flow-augmented temporal encoding.
> - Model sharing vs specialization: A single model is trained across staged tasks with shared weights and multi-task instruction fine-tuning.
> - Role of positional encoding: Coordinate/spatiotemporal encoding is a central variable with explicit ablation comparisons.

---

### 14. Final Classification

**Classification:** Multi-task, multi-domain (constrained)

Justification: The paper evaluates multiple tasks ("dense captioning (DC), visual QA and visual grounding (VG)") across multiple domains within vision, including "multiple 3D datasets" and a 4D dataset (Chat4D), as well as "2D, 3D and 4D vision-language data types" (Sections 4.1, 5.1). The scope is still constrained to structured 2D/3D/4D scene understanding tasks and datasets, rather than open-ended multi-domain learning.
