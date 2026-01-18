## 1. Basic Metadata

- Title: "Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation" (Title)
- Authors: "Jiahao Lu University of Science and Technology of China Jiacheng Deng* University of Science and Technology of China" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

It targets the problem that "3D instance segmentation aims to predict a set of object instances in a scene, representing them as binary foreground masks with corresponding semantic labels" by proposing "Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation" to improve relation modeling for point cloud instance segmentation (Abstract).

---

## 3. Tasks Evaluated

| Task name | Task type | Dataset(s) used | Domain | Evidence |
| --- | --- | --- | --- | --- |
| 3D point cloud instance segmentation | Segmentation | ScanNetV2; ScanNet++; ScanNet200; S3DIS | 3D point clouds; indoor scenes explicitly stated for Scan-Net++ and S3DIS | "3D instance segmentation aims to predict a set of object instances in a scene, representing them as binary foreground masks with corresponding semantic labels." (Abstract); "We conduct our experiments on ScanNetV2 [32], ScanNet++ [33], ScanNet200 [34], and S3DIS [35] datasets." (4.1. Experimental Setup); "Assuming that the input point cloud has N points, each point contains position (x, y, z), color (r, q, b)and normal  $(n_x, n_y, n_z)$  information." (3.1. Overview); "Scan-Net++ contains 460 high-resolution (sub-millimeter) indoor scenes with dense instance annotations across 84 unique instance categories." (4.1. Experimental Setup); "S3DIS is a largescale indoor dataset collected from six different areas, containing 272 scenes with 13 instance categories." (4.1. Experimental Setup) |

---

## 4. Domain and Modality Scope

- Single domain? Not explicitly stated; the evaluated datasets include indoor scenes ("Scan-Net++ contains 460 high-resolution (sub-millimeter) indoor scenes with dense instance annotations across 84 unique instance categories." (4.1. Experimental Setup); "S3DIS is a largescale indoor dataset collected from six different areas, containing 272 scenes with 13 instance categories." (4.1. Experimental Setup)).
- Multiple domains within the same modality? Multiple datasets are used within the point cloud modality ("We conduct our experiments on ScanNetV2 [32], ScanNet++ [33], ScanNet200 [34], and S3DIS [35] datasets." (4.1. Experimental Setup)).
- Multiple modalities? Not indicated; the input is a point cloud ("Assuming that the input point cloud has N points, each point contains position (x, y, z), color (r, q, b)and normal  $(n_x, n_y, n_z)$  information." (3.1. Overview)).
- Domain generalization or cross-domain transfer? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D point cloud instance segmentation | Not specified. | Not specified. | Not specified. | "We conduct our experiments on ScanNetV2 [32], ScanNet++ [33], ScanNet200 [34], and S3DIS [35] datasets." (4.1. Experimental Setup); "All the other hyperparameters are the same for all datasets." (4.1. Experimental Setup). No explicit weight-sharing statement. |

---

## 6. Input and Representation Constraints

- Fixed or variable input resolution? Not specified; the input is described as a point cloud with N points ("Assuming that the input point cloud has N points, each point contains position (x, y, z), color (r, q, b)and normal  $(n_x, n_y, n_z)$  information." (3.1. Overview)).
- Fixed patch size? Not specified.
- Fixed number of tokens? The instance query count K is set explicitly ("For hyperparameters, we tune K, r as 400, 3 respectively. Since ScanNet++ and ScanNet200 have more categories and instances, we set K as 500." (4.1. Experimental Setup)).
- Fixed dimensionality (e.g., strictly 2D)? The input point cloud uses 3D coordinates ("position (x, y, z)" (3.1. Overview)).
- Any padding or resizing requirements? Not specified.
- Voxelization / grid constraints? "Point clouds are voxelized with a size of 0.02m." (4.1. Experimental Setup).

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; the number of instance queries is set as K=400 or K=500 depending on dataset ("For hyperparameters, we tune K, r as 400, 3 respectively. Since ScanNet++ and ScanNet200 have more categories and instances, we set K as 500." (4.1. Experimental Setup)).
- Fixed or variable sequence length: Not specified.
- Attention type (global/windowed/hierarchical/sparse): Not specified; the paper introduces relation-aware self-attention ("we propose a relation-aware self-attention (RSA)." (3.5. Relation-aware Self-attention)).
- Computational cost mechanisms: "reduce computational and memory costs, we do not perform self-attention for self-updating  $F_{\mathrm{super}}$ . Furthermore, the superpoint refinement module is not applied at every decoder layer. Instead, we perform the refinement of  $F_{\mathrm{super}}$  every r layers to reduce computational resource consumption ." (3.4. Contrastive Learning-guided Superpoint Refinement Module).

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: The method constructs relative positional and geometric relationships between queries and applies sine-cosine encoding ("Positional Relative Relationship:" followed by relative position formulas and "Geometric Relative Relationship:" followed by relative scale formulas (3.5. Relation-aware Self-attention); "Then, following past methods, we use conventional sine-cosine encoding to increase the dimensionality of  $\mathfrak{T} \in \mathbb{R}^{K \times K \times 6d}$ ," (3.5. Relation-aware Self-attention)).
- Where it is applied: The relation embedding is added to attention logits as a bias term ("RSA(Q) = Softmax(\frac{QK^{T}}{\sqrt{C}} + R_q)V." (3.5. Relation-aware Self-attention)).
- Fixed across all experiments / modified per task / ablated: Not specified; no explicit comparison of positional encodings is described.

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed architectural assumption? The paper defines RSA with a fixed sine-cosine encoding step ("we use conventional sine-cosine encoding to increase the dimensionality of  $\mathfrak{T} \in \mathbb{R}^{K \times K \times 6d}$ ," (3.5. Relation-aware Self-attention)), but does not describe multiple positional encodings.
- Multiple positional encodings compared? Not specified.
- Claims that PE choice is not critical or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): "ScanNetV2 comprises 1,613 scenes with 18 instance categories, of which 1,201 scenes are used for training, 312 for validation, and 100 for testing." (4.1. Experimental Setup); "Scan-Net++ contains 460 high-resolution (sub-millimeter) indoor scenes with dense instance annotations across 84 unique instance categories." (4.1. Experimental Setup); "ScanNet200 uses the same point cloud data, but it enhances annotation diversity, covering 200 classes, 198 of which are instance classes." (4.1. Experimental Setup); "S3DIS is a largescale indoor dataset collected from six different areas, containing 272 scenes with 13 instance categories." (4.1. Experimental Setup).
- Performance gains attributed to scaling model size or data? Not claimed.
- Performance gains attributed to architecture or training tricks: "Due to our focus on modeling the internal relationships between the scene features and between the queries, our approach outperforms other transformer-based methods" (4.2. Comparison with existing methods); "This improvement can be attributed to the relation priors introduced by CLSR and RSA: contrastive learning provides relation priors for superpoints to guide feature aggregation, while RSA introduces position and geometric relation priors for query features, enhancing self-attention." (4.3. Ablation Studies).

---

## 11. Architectural Workarounds

- Adaptive superpoint aggregation to emphasize distinctive point features: "To emphasize distinctive and meaningful point features while diminishing the influence of unsuitable features, we design the adaptive superpoint aggregation module" (3.3. Adaptive Superpoint Aggregation Module).
- Dual-path refinement and contrastive supervision for superpoints: "This design, in conjunction with the original mask attention, forms a dual-path architecture, enabling direct communication between query and superpoint features. This approach accelerates the convergence speed of the iterative updates." (3.4. Contrastive Learning-guided Superpoint Refinement Module).
- Cost reduction by skipping self-attention on superpoints and applying refinement every r layers: "reduce computational and memory costs, we do not perform self-attention for self-updating  $F_{\mathrm{super}}$ . Furthermore, the superpoint refinement module is not applied at every decoder layer. Instead, we perform the refinement of  $F_{\mathrm{super}}$  every r layers to reduce computational resource consumption ." (3.4. Contrastive Learning-guided Superpoint Refinement Module).
- Relation-aware self-attention with explicit positional and geometric relations: "By obtaining the mask and its bounding box corresponding to each query, we can model the positional and geometric relationships between queries. Next, we embed these relationships into self-attention as embeddings." (1. Introduction).
- Fixed voxelization grid: "Point clouds are voxelized with a size of 0.02m." (4.1. Experimental Setup).
- Post-processing with NMS: "we also employ NMS [47] on the final output as a post-processing operation." (3.6. Model Training and Inference).

---

## 12. Explicit Limitations and Non-Claims

Not specified.

---

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Multiple point cloud datasets, with indoor scenes explicitly stated for Scan-Net++ and S3DIS.
- Task structure: Single task (3D point cloud instance segmentation) evaluated across datasets.
- Representation rigidity: 3D point clouds with specified point attributes and fixed voxel size; fixed query count K per dataset.
- Model sharing vs specialization: Weight sharing across datasets is not specified; training appears dataset-specific with shared hyperparameter settings.
- Role of positional encoding: RSA adds relative positional/geometric embeddings with sine-cosine encoding as an attention bias; no alternative PE comparisons reported.

---

### 14. Final Classification

**Single-task, single-domain**

The paper evaluates one task, as "3D instance segmentation aims to predict a set of object instances in a scene, representing them as binary foreground masks with corresponding semantic labels" (Abstract), and all evaluations use point cloud datasets ("We conduct our experiments on ScanNetV2 [32], ScanNet++ [33], ScanNet200 [34], and S3DIS [35] datasets." (4.1. Experimental Setup)). The domains described are consistent with indoor 3D scenes ("Scan-Net++ contains 460 high-resolution (sub-millimeter) indoor scenes..." and "S3DIS is a largescale indoor dataset..." (4.1. Experimental Setup)), and there is no claim of cross-domain transfer.
