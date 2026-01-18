## 1. Basic Metadata

- Title: "POS-BERT: Point Cloud One-Stage BERT Pre-Training" (Title)
- Authors: Authors not specified.
- Year: "POS-BERT (our)       | 2022 | point | <b>92.1</b> %" (Table 1: Classification results with linear SVM on ModelNet40)
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"Inspired by BERT and MoCo, we propose POS-BERT, a one-stage BERT pre-training method for point clouds." (Abstract)

## 3. Tasks Evaluated

- Task name: Linear SVM classification (ModelNet40)
  - Task type: Classification
  - Dataset(s) used: ShapeNet (pre-training), ModelNet40 (evaluation)
  - Domain: 3D point clouds (synthetic CAD models)
  - Evidence: "Linear SVM classification task has become a classic task to evaluate self-supervised point cloud representation learning." (Section 5.1 Linear SVM Classification) "pre-trained the model on ShapeNet and tested it on the ModelNet40." (Section 5.1 Linear SVM Classification) "ShapeNet contains 57448 CAD models, with a total of 55 categories." (Section 4.2 Dataset) "ModelNet40 contains 12,331 handmade CAD models of from 40 categories and is widely used for point cloud classification tasks." (Section 4.2 Dataset)

- Task name: 3D object classification on synthetic data (ModelNet40 fine-tuning)
  - Task type: Classification
  - Dataset(s) used: ShapeNet (pre-training), ModelNet40 (fine-tuning)
  - Domain: 3D point clouds (synthetic CAD models)
  - Evidence: "**3D Object Classification on Synthetic Data** To test whether POS-BERT can help boost downstream tasks. We first performed fine-tuning experiments on point cloud classification tasks using a pretraining model." (Section 5.2 Downstream Tasks) "Pretrain stands for pre-training the model on ShapeNet and then fine-tune the network on ModelNet40." (Section 5.2 Downstream Tasks)

- Task name: Few-shot classification (Fewshot-ModelNet40)
  - Task type: Classification
  - Dataset(s) used: Fewshot-ModelNet40 (derived from ModelNet40)
  - Domain: 3D point clouds (synthetic CAD models)
  - Evidence: "Following the work of Yu et al. [40], we generated a Fewshot-ModelNet40 dataset based on ModelNet40." (Section 4.2 Dataset) "**Few-shot Classification** To demonstrate that our pre-training model can learn quickly from few-shot samples, we conduct experiment on the Few-shot ModelNet40 dataset." (Section 5.2 Downstream Tasks)

- Task name: 3D object classification on real-world data (ScanObjectNN)
  - Task type: Classification
  - Dataset(s) used: ScanObjectNN (OBJ-BG, OBJ-ONLY, PB-T50-RS)
  - Domain: 3D point clouds (real-world scanned data)
  - Evidence: "**3D Object Classification on Real-world Data** In this experiment, we aim to explore whether the knowledge POS-BERT learns from ShapNet can be transferred to real-world data. We conduct experiments on three variants of ScanObjectNN [60] dataset, including OBJ-BG, OBJ-ONLY, and PB-T50-RS." (Section 5.2 Downstream Tasks) "SacnObjectNN is a 3D point cloud classification dataset derived from real-world scanned data." (Section 4.2 Dataset)

- Task name: Part segmentation (ShapeNetPart)
  - Task type: Segmentation
  - Dataset(s) used: ShapeNetPart
  - Domain: 3D point clouds (synthetic CAD models)
  - Evidence: "**Part Segmentation** In this section, we explore how the pre-training model performs in the pre-point classification. We experimented on ShapeNetPart, a benchmark dataset commonly used in point cloud segmentation tasks." (Section 5.2 Downstream Tasks)

## 4. Domain and Modality Scope

- Evaluation performed on a single domain? Multiple domains within the same modality? Multiple modalities?
  - Multiple domains within the same modality (3D point clouds): synthetic CAD datasets and real-world scanned data. Evidence: "ShapeNet contains 57448 CAD models" (Section 4.2 Dataset) and "ModelNet40 contains 12,331 handmade CAD models" (Section 4.2 Dataset) versus "SacnObjectNN is a 3D point cloud classification dataset derived from real-world scanned data." (Section 4.2 Dataset).
  - Multiple modalities? Not stated; evaluations are on point clouds only.
- Does the paper claim domain generalization or cross-domain transfer?
  - Yes, synthetic-to-real transfer is claimed: "we aim to explore whether the knowledge POS-BERT learns from ShapNet can be transferred to real-world data." (Section 5.2 Downstream Tasks)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Linear SVM classification (ModelNet40) | Yes (pretraining model used to extract features) | No | Yes (linear SVM) | "Using the pretraining model without any fine-tuning to extract features and train linear SVM on ModelNet40" (Abstract) and "we use our pre-training model to extract the features of each point cloud, then trained a simple linear Support Vector Machine (SVM)" (Section 5.1 Linear SVM Classification) |
| 3D object classification on synthetic data (ModelNet40 fine-tuning) | Yes (pretrained initialization) | Yes | Yes (classification head) | "Pretrain stands for pre-training the model on ShapeNet and then fine-tune the network on ModelNet40." (Section 5.2 Downstream Tasks) and "We use a fully connected MLP network that combines ReLU, BN, and Dropout operations as the classification head." (Section 4.1 Implementation) |
| Few-shot classification (Fewshot-ModelNet40) | Yes (pre-training model referenced) | Not specified. | Not specified. | "To demonstrate that our pre-training model can learn quickly from few-shot samples, we conduct experiment on the Few-shot ModelNet40 dataset." (Section 5.2 Downstream Tasks) |
| 3D object classification on real-world data (ScanObjectNN) | Yes (transfer from ShapeNet implied) | Not specified. | Not specified. | "we aim to explore whether the knowledge POS-BERT learns from ShapNet can be transferred to real-world data." (Section 5.2 Downstream Tasks) |
| Part segmentation (ShapeNetPart) | Yes (pre-training model referenced) | Not specified. | Yes (segmentation head) | "we explore how the pre-training model performs in the pre-point classification." (Section 5.2 Downstream Tasks) and "Different from the classification task, the segmentation task needs to predict pre-point labels... Finally, MLP is used to map the features to the segmentation label space." (Section 4.1 Implementation) |

## 6. Input and Representation Constraints

- Dimensionality / modality: "the raw point clouds  $P \in \mathbb{R}^{N \times 3}$" (Section 3 Method).
- Fixed patch size: "we divide a given global/local point cloud P into local patches with a fixed number of K points." (Section 3.1 Point2Patch Embedding and Encoder Architecture)
- Number of tokens (patches) tied to input size: "we first calculate the number of patches  $Q = \operatorname{ceil}(N/K)$" (Section 3.1 Point2Patch Embedding and Encoder Architecture)
- Fixed input sizes used in experiments:
  - "sample 2048 points from each CAD model surface." (Section 4.2 Dataset, ShapeNet)
  - "we use the farthest point sampling algorithm to select 64 group center points, and divide 2048 points into 64 groups, where each group contains 32 points." (Section 4.2 Dataset, ShapeNet)
  - "we follow Yu et al. to sample 8192 points from each CAD model surface." (Section 4.2 Dataset, ModelNet40)
  - "we randomly select 2048 points as input." (Section 4.2 Dataset, ShapeNetPart)
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; token count is defined by patching, "$Q = \operatorname{ceil}(N/K)$" with input tokens "$T_0=\{t_0,t_1,\ldots,t_Q\}$" (Section 3.1 Point2Patch Embedding and Encoder Architecture).
- Fixed or variable sequence length: Variable in general via $Q = \operatorname{ceil}(N/K)$, but experiments fix N and K (e.g., 2048 points into 64 groups of 32). Evidence: "we first calculate the number of patches  $Q = \operatorname{ceil}(N/K)$" (Section 3.1) and "divide 2048 points into 64 groups, where each group contains 32 points." (Section 4.2 Dataset).
- Attention type: Global (standard full self-attention). Evidence: "We used a standard transformer as the Encoder backbone, which consists of a series of stacked multihead self-attention layers" (Section 3.1 Point2Patch Embedding and Encoder Architecture).
- Mechanisms to manage computational cost: Patch tokenization to reduce O(N^2) complexity. Evidence: "Because the complexity of transformer is  $O(N^2)$ ... Following Point-BERT, we divide a given global/local point cloud P into local patches with a fixed number of K points." (Section 3.1 Point2Patch Embedding and Encoder Architecture)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Absolute (learned MLP on patch center coordinates).
  - Evidence: "the center point position embedding  $pos = mlp(c_i)$  corresponding to patch tokens is added to  $m_t$ ,  $c_i$  represents the xyz coordinate of the patch center point." (Section 3.2 Mask Patch Modeling)
- Where it is applied: Input-level addition to masked patch tokens.
  - Evidence: "position information is added to the corresponding masked patches" (Section 3 Method) and "the center point position embedding  $pos = mlp(c_i)$  corresponding to patch tokens is added to  $m_t$" (Section 3.2 Mask Patch Modeling)
- Fixed across all experiments / modified per task / ablated: Not specified.

## 9. Positional Encoding as a Variable

- Role: Fixed architectural assumption (used, not studied as a variable). Evidence: "the center point position embedding  $pos = mlp(c_i)$  corresponding to patch tokens is added to  $m_t$" (Section 3.2 Mask Patch Modeling).
- Multiple positional encodings compared? Not specified.
- Any claim that PE choice is not critical? Not specified.

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): "ShapeNet contains 57448 CAD models, with a total of 55 categories." (Section 4.2 Dataset) "ModelNet40 contains 12,331 handmade CAD models" (Section 4.2 Dataset) "SacnObjectNN... contains 2902 point clouds from 15 categories." (Section 4.2 Dataset) "ShapeNetPart contains 16811 objects from 16 categories." (Section 4.2 Dataset)
- Primary source of gains (scaling vs architecture/training): The paper attributes gains to architectural/training choices rather than scaling. Evidence: "This result fully shows that our Momentum Encoder can provide more meaningful supervision representation for masked patches." (Section 5.1 Linear SVM Classification) and "Pre-training with masking patch modeling alone is difficult to obtain high-level semantic information. The best results are obtained when masking patch modeling and contrastive learning work together." (Section 5.3 Ablation study)
- Explicit scaling claims (model or data): Not specified.

## 11. Architectural Workarounds

- Patch-based tokenization to reduce quadratic attention cost: "Because the complexity of transformer is  $O(N^2)$ ... we divide a given global/local point cloud P into local patches with a fixed number of K points." (Section 3.1 Point2Patch Embedding and Encoder Architecture)
- Patch construction with sampling and neighborhood grouping: "use farthest point sampling (FPS) algorithm to sample the center point  $c_i$  of each patch. The k-nearest neighbor algorithm is used to obtain K neighbors" (Section 3.1 Point2Patch Embedding and Encoder Architecture)
- Global/local cropping for contrastive supervision: "the global point cloud set  $P_g$  and the local point cloud set  $P_l$  are obtained by cropping the raw point clouds" (Section 3 Method)
- Dynamic tokenizer via momentum encoder: "we propose a dynamically updated tokenizer, which is implemented by momentum Encoder." (Section 3.3 Dynamic Tokenizer by Momentum Encoder)
- Task-specific heads: "We use a fully connected MLP network that combines ReLU, BN, and Dropout operations as the classification head." (Section 4.1 Implementation) and "Finally, MLP is used to map the features to the segmentation label space." (Section 4.1 Implementation)

## 12. Explicit Limitations and Non-Claims

Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: 3D point clouds only, spanning synthetic CAD datasets and real-world scanned data.
> - Task structure: Multiple downstream tasks evaluated separately (linear SVM classification, fine-tuned classification, few-shot classification, part segmentation).
> - Representation rigidity: Fixed point counts per dataset (e.g., 2048 or 8192 points) and fixed patch size K with Q = ceil(N/K).
> - Model sharing vs specialization: Single pretrained model reused for evaluation and fine-tuning, with separate classification and segmentation heads.
> - Role of positional encoding: Simple absolute center-point MLP embedding added to masked tokens; no ablations reported.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates multiple tasks (classification variants and part segmentation) and uses both synthetic CAD datasets and real-world scanned data within the point cloud modality (e.g., "ModelNet40 contains 12,331 handmade CAD models" and "SacnObjectNN is a 3D point cloud classification dataset derived from real-world scanned data"). It explicitly frames cross-domain transfer from ShapeNet to ScanObjectNN ("knowledge POS-BERT learns from ShapNet can be transferred to real-world data"), but remains constrained to 3D point cloud inputs and standard transformer architecture.
