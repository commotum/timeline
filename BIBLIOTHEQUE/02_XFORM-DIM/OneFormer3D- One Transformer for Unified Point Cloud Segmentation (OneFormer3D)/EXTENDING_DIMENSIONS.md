## 1. Basic Metadata

- Title: OneFormer3D: One Transformer for Unified Point Cloud Segmentation. Quote (Title block): "OneFormer3D: One Transformer for Unified Point Cloud Segmentation"
- Authors: Maxim Kolodiazhnyi, Anna Vorontsova, Anton Konushin, Danila Rukhovich (Samsung Research). Quote (Title block): "Maxim Kolodiazhnyi, Anna Vorontsova, Anton Konushin, Danila Rukhovich Samsung Research"
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The Abstract states that the paper presents "a unified, simple, and effective model addressing all these tasks jointly" for "semantic, instance, and panoptic segmentation of 3D point clouds" (Abstract).

---

## 3. Tasks Evaluated

- Task name: Semantic segmentation (3D point clouds)
- Task type: Segmentation
- Dataset(s) used: ScanNet, ScanNet200, S3DIS
- Domain: 3D point clouds (indoor scenes)
- Evidence: "Semantic segmentation outputs a mask for each semantic category, so that each point in a point cloud gets assigned with a semantic label." (1. Introduction) "The experiments are conducted on ScanNet [8], ScanNet200 [28], and S3DIS [1] datasets." (4.1. Experimental Settings) "We compare our OneFormer3D with previous art on three indoor benchmarks: ScanNet [8], S3DIS [1], and Scan-Net200 [28]" (4.2. Comparison to Prior Work)

- Task name: Instance segmentation (3D point clouds)
- Task type: Segmentation
- Dataset(s) used: ScanNet, ScanNet200, S3DIS
- Domain: 3D point clouds (indoor scenes)
- Evidence: "Instance segmentation returns a set of masks of individual objects; since some regions cannot be treated as an distinguishable object but rather serve as a background (like a floor or a ceiling), only a part of points in a point cloud is being labeled." (1. Introduction) "The experiments are conducted on ScanNet [8], ScanNet200 [28], and S3DIS [1] datasets." (4.1. Experimental Settings) "We compare our OneFormer3D with previous art on three indoor benchmarks: ScanNet [8], S3DIS [1], and Scan-Net200 [28]" (4.2. Comparison to Prior Work)

- Task name: Panoptic segmentation (3D point clouds)
- Task type: Segmentation
- Dataset(s) used: ScanNet, ScanNet200, S3DIS
- Domain: 3D point clouds (indoor scenes)
- Evidence: "Semantic, instance, and panoptic segmentation of 3D point clouds have been addressed using task-specific models of distinct design." (Abstract) "The experiments are conducted on ScanNet [8], ScanNet200 [28], and S3DIS [1] datasets." (4.1. Experimental Settings) "We compare our OneFormer3D with previous art on three indoor benchmarks: ScanNet [8], S3DIS [1], and Scan-Net200 [28]" (4.2. Comparison to Prior Work)

- Task name: 3D object detection
- Task type: Detection
- Dataset(s) used: ScanNet (validation split)
- Domain: 3D point clouds (indoor scenes)
- Evidence: "Besides, we adopt OneFormer3D to 3D object detection by enclosing predicted 3D instances with tight axis-aligned 3D bounding boxes." (4.2. Comparison to Prior Work) "Table 1. Comparison of existing 3D object detection methods on the ScanNet validation split." (Table 1 caption)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Single domain (indoor 3D point clouds). Evidence: "3D point cloud segmentation is the task of grouping points into meaningful segments." (1. Introduction) "We compare our OneFormer3D with previous art on three indoor benchmarks: ScanNet [8], S3DIS [1], and Scan-Net200 [28]" (4.2. Comparison to Prior Work)
- Multiple domains within the same modality: Not indicated; all listed datasets are indoor 3D point clouds (same modality). Evidence: "The experiments are conducted on ScanNet [8], ScanNet200 [28], and S3DIS [1] datasets." (4.1. Experimental Settings)
- Multiple modalities: Not indicated; the method operates on 3D point clouds and explicitly avoids extra RGB images. Evidence: "On the contrary, our OneFormer3D does not require additional RGB data to achieve state-of-the-art panoptic segmentation quality." (2.1. 3D Panoptic Segmentation)
- Domain generalization or cross-domain transfer: Not claimed. (They mention pretraining across real and synthetic data but do not claim cross-domain transfer.) Evidence: "benefits from using a larger amount of training data exceed the possible negative effect of a domain gap" (4.3. Ablation Studies)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Semantic segmentation | Yes | Not specified | Yes (semantic queries/kernels) | "This paper presents a unified, simple, and effective model addressing all these tasks jointly." (Abstract) "A query decoder takes K_ins + K_sem queries as inputs and transforms them into K_ins + K_sem kernels." (3.2. Query Decoder) |
| Instance segmentation | Yes | Not specified | Yes (instance queries/kernels) | "Such a design enables training a model end-to-end in a single run, so that it achieves top performance on all three segmentation tasks simultaneously." (Abstract) "A query decoder takes K_ins + K_sem queries as inputs and transforms them into K_ins + K_sem kernels." (3.2. Query Decoder) |
| Panoptic segmentation | Yes | Not specified | Uses instance + semantic outputs | "Trained only once on a panoptic dataset, OneFormer3D consistently outperforms existing segmentation approaches" (5. Conclusion) "Panoptic prediction is obtained from instance and semantic outputs." (3.4. Inference) |
| 3D object detection | Yes (derived from OneFormer3D outputs) | No extra training | Not specified (boxes from predicted instances) | "we adopt OneFormer3D to 3D object detection by enclosing predicted 3D instances with tight axis-aligned 3D bounding boxes." (4.2. Comparison to Prior Work) "setting a new state-of-the-art in 3D object detection with 65.1 mAP<sub>50</sub> with no extra training." (4.2. Comparison to Prior Work) |

---

## 6. Input and Representation Constraints

- Input point representation is fixed to 6D per point (RGB + XYZ): "Assuming that an input point cloud contains N points, the input can be formulated as P in R^{N x 6}. Each 3D point is parameterized with three colors r, g, b, and three coordinates x, y, z." (3.1. Backbone and Pooling)
- Voxelization is applied, with fixed voxel sizes per dataset: "Following [6], we voxelize point cloud" (3.1. Backbone and Pooling) and "On ScanNet and ScanNet200, we apply graph-based superpoint clusterization [18] and use a voxel size of 2cm. On S3DIS, voxel size is set to 5cm due to larger scenes." (4.1. Experimental Settings)
- Pooling constrains representation to superpoints or voxels: "superpoint features S in R^{M x C} are obtained via average pooling of point-wise features" and "In a voxel pooling scenario, we pool backbone features w.r.t. voxel grid." (3.1. Backbone and Pooling)
- Fixed semantic category ordering for semantic queries: "K_sem semantic queries correspond to ground truth masks of K_sem semantic categories given in a fixed order" (3.3. Training)
- The representation is aggressively downsampled for efficiency: "This procedure transforms an input point cloud comprised of millions of points into only hundreds of superpoints or thousands of voxels" (3.1. Backbone and Pooling)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; the decoder uses a fixed number of queries K_ins + K_sem and superpoint features, but no explicit maximum is given. Evidence: "A query decoder takes K_ins + K_sem queries as inputs" (3.2. Query Decoder) and "we suppose that there are M superpoints in an input point cloud." (3.1. Backbone and Pooling)
- Fixed or variable sequence length: Not specified; input size is variable in N points and M superpoints. Evidence: "Assuming that an input point cloud contains N points" and "we suppose that there are M superpoints in an input point cloud." (3.1. Backbone and Pooling)
- Attention type: Self-attention on queries with cross-attention to superpoint features; no windowing/sparsity described. Evidence: "six sequential transformer decoder layers employ self-attention on queries and cross-attention with keys and values from superpoint features." (3.2. Query Decoder)
- Mechanisms to manage computational cost: Flexible pooling reduces token count. Evidence: "This procedure transforms an input point cloud comprised of millions of points into only hundreds of superpoints or thousands of voxels, which significantly reduces the computational cost of subsequent processing." (3.1. Backbone and Pooling)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Not specified.
- Where it is applied: Not specified.
- Fixed/modified/ablated: Not specified.

---

## 9. Positional Encoding as a Variable

- Treated as a core research variable or fixed assumption: Not specified.
- Multiple positional encodings compared: Not specified.
- Claims that PE choice is not critical or secondary: Not specified.

---

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): "ScanNet [8] contains 1613 scans divided into training, validation, and testing splits of 1201, 312, and 100 scans, respectively." (4.1. Experimental Settings) "The S3DIS dataset [1] features 272 scenes within 6 large areas." (4.1. Experimental Settings) "Structured3D [50] dataset for pretraining, which is an order of magnitude larger than ScanNet, with as many as 21835 scenes." (4.3. Ablation Studies)
- Evidence that scaling data helps: "benefits from using a larger amount of training data exceed the possible negative effect of a domain gap: the best results are achieved with pretraining on a mixture of real and synthetic data" (4.3. Ablation Studies)
- Evidence that architectural/training tricks drive gains: "the synergy of these two modifications allows for the state-of-the-art results" (4.3. Ablation Studies) and "we refer to it as our disentangled matching" with "O(K_ins) complexity" (3.3. Training)

---

## 11. Architectural Workarounds

- Sparse 3D U-Net with voxelization for feature extraction: "Following [6], we voxelize point cloud, and use a U-Net-like backbone composed of sparse 3D convolutions to extract point-wise features" (3.1. Backbone and Pooling)
- Flexible pooling (superpoints or voxels) to reduce computation: "This procedure transforms an input point cloud comprised of millions of points into only hundreds of superpoints or thousands of voxels, which significantly reduces the computational cost of subsequent processing." (3.1. Backbone and Pooling)
- Parallel semantic and instance queries to unify tasks: "we add semantic queries in parallel with instance queries in a transformer decoder to unify predicting semantic and instance segmentation masks." (1. Introduction)
- Query selection to stabilize training: "we aim to close this gap with a simplified version of query selection adapted for 3D data" (3.2. Query Decoder)
- Disentangled matching to avoid Hungarian complexity: "we perform a simple trick that eliminates the need for resource-exhaustive Hungarian matching" and "Our disentangled matching is notably more efficient, having a O(K_ins) complexity." (3.3. Training)

---

## 12. Explicit Limitations and Non-Claims

- Limitations or future work: Not specified.
- Explicit non-claims about scope (open-world learning, unrestrained multi-task, etc.): Not specified.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Indoor 3D point clouds only, evaluated on ScanNet, ScanNet200, and S3DIS.
> - Task structure: Multi-task segmentation (semantic, instance, panoptic) plus a derived 3D detection evaluation, all within the same 3D point cloud setting.
> - Representation rigidity: Fixed point features (RGB+XYZ), voxelization with dataset-specific voxel sizes, and superpoint/voxel pooling.
> - Model sharing vs specialization: Single unified model trained once for all segmentation tasks, with separate semantic/instance query sets.
> - Role of positional encoding: Not specified in the provided text.

---

### 14. Final Classification

Multi-task, single-domain. The paper explicitly targets "semantic, instance, and panoptic segmentation of 3D point clouds" and trains "a unified, simple, and effective model addressing all these tasks jointly" (Abstract). Evaluation is on "three indoor benchmarks: ScanNet [8], S3DIS [1], and Scan-Net200 [28]" (4.2. Comparison to Prior Work), and no cross-domain or multi-modality claim is made.
