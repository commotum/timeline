## 1. Basic Metadata

- Title: "ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention" (Title)
- Authors: "Chenhang He<sup>1</sup>, Ruihuang Li<sup>1,2</sup>, Guowen Zhang<sup>1</sup>, and Lei Zhang<sup>1,2</sup>" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.


## 2. One-Sentence Contribution Summary

It claims the overhead from window-based voxel transformers because "existing methods group the voxels in each window into fixed-length sequences through extensive sorting and padding operations" and thus "we introduce ScatterFormer, which to the best of our knowledge, is the first to directly apply attention to voxels across different windows as a single sequence" (Abstract).


## 3. Tasks Evaluated

Task 1
Task name: 3D object detection (point clouds)
Task type: Detection
Dataset(s) used: Waymo Open Dataset (WOD); NuScenes
Domain: LiDAR point clouds (autonomous driving/outdoor)
Quotes: "In the field of 3D object detection, the use of point clouds has become increasingly popular, especially for providing accurate and reliable perception results in autonomous systems." (1 Introduction) "Waymo Open Dataset (WOD). This dataset contains 230,000 annotated samples split into 160,000 for training, 40,000 for validation, and 30,000 for testing. It uses two metrics for 3D object detection: mean average precision (mAP) and mAP weighted by heading accuracy (mAPH)" (4.1 Datasets and Evaluation Metrics) "NuScenes. This dataset comprises 40,000 annotated samples, with 28,000 for training, 6,000 for validation, and 6,000 for testing." (4.1 Datasets and Evaluation Metrics) "point clouds obtained from LiDAR are often sparse and nonuniformly distributed" (1 Introduction).


## 4. Domain and Modality Scope

- Single domain: Yes; LiDAR point clouds for 3D object detection in autonomous systems ("In the field of 3D object detection, the use of point clouds has become increasingly popular" and "point clouds obtained from LiDAR are often sparse and nonuniformly distributed" (1 Introduction)).
- Multiple domains within the same modality: Not claimed; evaluation is reported on two LiDAR point cloud datasets ("Waymo Open Dataset (WOD). This dataset contains 230,000 annotated samples" and "NuScenes. This dataset comprises 40,000 annotated samples" (4.1 Datasets and Evaluation Metrics)).
- Multiple modalities: No; only LiDAR point clouds are described ("point clouds obtained from LiDAR are often sparse and nonuniformly distributed" (1 Introduction)).
- Domain generalization or cross-domain transfer: Not claimed.


## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D object detection (LiDAR point clouds) | Not specified. | Not specified. | Not specified. | "To construct ScatterFormer, we set the voxel size to (0.32m, 0.32m, 0.1875m) for the Waymo dataset and (0.3m, 0.3m, 8m) for the NuScenes dataset." (4.2 Implementation Details); "ScatterFormer is trained for 24 epochs with a learning rate of 0.006 on Waymo Dataset and 20 epochs with a learning rate of 0.004 on NuScenes Dataset." (4.2 Implementation Details) |


## 6. Input and Representation Constraints

- Voxelized 3D input: "It begins with the input point clouds, which are voxelized and transformed into high-dimensional embeddings using a VFE layer [60]." (3 Method)
- Fixed voxel size per dataset: "we set the voxel size to (0.32m, 0.32m, 0.1875m) for the Waymo dataset and (0.3m, 0.3m, 8m) for the NuScenes dataset." (4.2 Implementation Details)
- Fixed window size per dataset: "The window sizes (S_w, S_h) for the two datasets are set to (12, 12) and (20, 20), respectively." (4.2 Implementation Details)
- Variable token counts per window: "the number of features grouped by windows can vary significantly" and "In this paper, we delve into the window-based voxel transformer where the voxels grouped by windows form variable-length sequences  $\{X_1 \in \mathbb{R}^{n_1 \times d}, X_2 \in \mathbb{R}^{n_2 \times d}, ..., X_k \in \mathbb{R}^{n_k \times d}\}$ ." (1 Introduction)
- No fixed-length voxel sets required: "ScatterFormer stands out by not requiring voxel features to be organized into fixed-length sets [8, 23, 47]" (3 Method)
- Explicit zero-filling in BEV: "placing them back to their spatial locations and filling the unoccupied positions with zeros." (3.4 Detection Head and Loss)


## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Variable; "the number of features grouped by windows can vary significantly" and "In this paper, we delve into the window-based voxel transformer where the voxels grouped by windows form variable-length sequences  $\{X_1 \in \mathbb{R}^{n_1 \times d}, X_2 \in \mathbb{R}^{n_2 \times d}, ..., X_k \in \mathbb{R}^{n_k \times d}\}$ ." (1 Introduction).
- Attention type: Window-based scattered linear attention; "In this paper, we delve into the window-based voxel transformer where the voxels grouped by windows form variable-length sequences  $\{X_1 \in \mathbb{R}^{n_1 \times d}, X_2 \in \mathbb{R}^{n_2 \times d}, ..., X_k \in \mathbb{R}^{n_k \times d}\}$ ." and "the SLA module treats the voxels of the entire scene into a single sequence and processes them directly without padding voxels" (1 Introduction).
- Cost-management mechanisms: "Scattered Linear Attention (SLA) module, which leverages the pre-computation of key-value pairs in linear attention" and "a chunkwise algorithm that reduces the SLA module's latency to less than 1 millisecond" (Abstract); "we divide the voxel sequence into multiple chunks and loaded them into the shared memory (SRAM) of the GPU" (3.2 Scattered Linear Attention).


## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: "The backbone comprises a Conditional Position Encoding (CPE) and six transformer blocks." (Fig. 2 caption) and "These embeddings are then processed through Conditional Positional Encoding (CPE) using a shallow convolutional network [6]." (3 Method) Specific type (absolute/relative/RoPE/axial/bias-based) not specified.
- Where applied: Input before the backbone; "These embeddings are then processed through Conditional Positional Encoding (CPE) using a shallow convolutional network [6]. The encoded features enter the ScatterFormer backbone, consisting of six ScatterFormer blocks." (3 Method).
- Fixed across experiments or modified: CPE is ablated; "removing Conditional Position Encoding (CPE) module." (4.4 Ablation Study)


## 9. Positional Encoding as a Variable

- Treated as a core research variable or fixed assumption: It is tested as a component via ablation rather than presented as a core variable; "Table 4 shows four different configurations of ScatterFormer: (a) removing the Scattered Linear Attention module (SLA), (b) removing the Cross Window Interaction (CWI) module, (c) replacing the CWI with Shifted Window (SW) approach, and (d) removing Conditional Position Encoding (CPE) module." (4.4 Ablation Study)
- Multiple positional encodings compared: Not stated; only removal of CPE is reported (4.4 Ablation Study).
- Claim that PE choice is not critical: Not claimed.


## 10. Evidence of Constraint Masking

- Model size/configuration: "We configure our attention module to have 4 heads with a dimensionality of 128." (4.2 Implementation Details)
- Dataset sizes: "Waymo Open Dataset (WOD). This dataset contains 230,000 annotated samples" and "NuScenes. This dataset comprises 40,000 annotated samples" (4.1 Datasets and Evaluation Metrics).
- Performance gains attributed to architecture: "This improvement is attributed to our use of Linear Attention, which eliminates the need for extensive voxel sorting and padding operations." (4.3 Comparison with State-of-the-Arts)
- Efficiency claim from architectural change: "we propose a chunkwise algorithm that reduces the SLA module's latency to less than 1 millisecond on moderate GPUs." (Abstract)
- Scaling model size or data as primary driver: Not claimed.


## 11. Architectural Workarounds

- Window-based grouping with variable-length sequences to handle sparsity: "In this paper, we delve into the window-based voxel transformer where the voxels grouped by windows form variable-length sequences  $\{X_1 \in \mathbb{R}^{n_1 \times d}, X_2 \in \mathbb{R}^{n_2 \times d}, ..., X_k \in \mathbb{R}^{n_k \times d}\}$ ." (1 Introduction).
- Scattered Linear Attention to avoid padding/sorting overhead: "the SLA module treats the voxels of the entire scene into a single sequence and processes them directly without padding voxels" (1 Introduction).
- Chunkwise GPU algorithm for memory/I-O efficiency: "we divide the voxel sequence into multiple chunks and loaded them into the shared memory (SRAM) of the GPU" (3.2 Scattered Linear Attention).
- Cross-Window Interaction to avoid window shifting: "we propose a cross-window interaction module that improves the locality and connectivity of voxel features across different windows, eliminating the need for extensive window shifting." (Abstract)
- Hierarchical downsampling and BEV conversion for detection: "After three ScatterFormer blocks, the voxel features are downsampled via a sparse convolutional layer. The downsampled features are then converted into pillar features [47], generating compact BEV features for bounding-box prediction." (3 Method)


## 12. Explicit Limitations and Non-Claims

- Limitation (deployment): "ScatterFormer relies on our customized operators, which have not yet been implemented as plugins in TensorRT. Therefore, deploying ScatterFormer on invehicle devices will require additional engineering efforts." (4.6 Limitations)
- Future optimization direction: "ScatterFormer can be optimized by dynamically partitioning matrices according to different GPU architectures to leverage TensorCore for hardware acceleration." (4.6 Limitations)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.


### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Single modality (LiDAR point clouds) across two autonomous-driving datasets (Waymo, NuScenes).
- Task structure: Only 3D object detection is evaluated.
- Representation rigidity: Fixed voxel sizes and window sizes per dataset; variable-length voxel sequences per window.
- Model sharing vs specialization: Dataset-specific training configurations are described; shared-weight or joint training is not specified.
- Role of positional encoding: CPE is used before the backbone and ablated; no alternative PE comparisons are reported.


### 14. Final Classification

**Single-task, single-domain.** The paper targets "3D object detection using point clouds" and evaluates on LiDAR point cloud datasets (Waymo and NuScenes) within the same application domain (1 Introduction; 4.1 Datasets and Evaluation Metrics; 5 Conclusion). It does not report additional tasks or modalities, and no cross-domain transfer is claimed.
