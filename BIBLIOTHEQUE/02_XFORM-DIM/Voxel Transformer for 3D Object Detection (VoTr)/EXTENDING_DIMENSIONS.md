## 1. Basic Metadata

- Title: "Voxel Transformer for 3D Object Detection" (title)
- Authors: "Jiageng Mao  $^{1*}$  Yujing Xue  $^{2*}$  Minzhe Niu  $^3$  Haoyue Bai  $^4$  Jiashi Feng  $^2$  Xiaodan Liang  $^5$  Hang Xu  $^{3\dagger}$  Chunjing Xu  $^3$" (title page)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes a "voxel-based Transformer backbone for 3D object detection from point clouds" to address limited receptive fields by enabling long-range voxel relationships via self-attention. (Abstract)

## 3. Tasks Evaluated

**Task 1**
- Task name: 3D vehicle detection (Waymo Open)
- Task type: Detection
- Dataset(s) used: Waymo Open dataset
- Domain: Point clouds / LiDAR (autonomous driving)
- Evidence: "We evaluate Voxel Transformer on the commonly used Waymo Open dataset [30]" (4. Experiments); "Performance comparison on the Waymo Open Dataset with 202 validation sequences for the vehicle detection." (Table 1 caption); "The Waymo Open Dataset contains 1000 sequences in total, including 798 sequences (around 158k point cloud samples) in the training set and 202 sequences (around 40k point cloud samples) in the validation set." (4.1. Experimental Setup)

**Task 2**
- Task name: 3D car detection (KITTI)
- Task type: Detection
- Dataset(s) used: KITTI dataset
- Domain: Point clouds / LiDAR (autonomous driving)
- Evidence: "We evaluate Voxel Transformer on the commonly used Waymo Open dataset [30] and the KITTI [8] dataset." (4. Experiments); "Performance comparison on the KITTI *test* set with AP calculated by 40 recall positions for the car category." (Table 2 caption); "The KITTI dataset contains 7481 training samples and 7518 test samples" (4.1. Experimental Setup)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple datasets within the same modality (point clouds). Evidence: "3D object detection from point clouds" (Abstract); "We evaluate Voxel Transformer on the commonly used Waymo Open dataset [30] and the KITTI [8] dataset." (4. Experiments)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D vehicle detection (Waymo Open) | Not specified. Training described separately per dataset. | Not specified. | Not specified. | "On the Waymo Open dataset, we uniformly sample 20% frames for training and use the full validation set for evaluation" (Training and Inference Details, 4.1) |
| 3D car detection (KITTI) | Not specified. Training described separately per dataset. | Not specified. | Not specified. | "On the KITTI dataset, VoTr-SSD and VoTr-TSD are trained with the batch size 32 and 16 respectively, and with the learning rate 0.01 for 80 epochs" (Training and Inference Details, 4.1) |

## 6. Input and Representation Constraints

- Voxelization into a dense grid and sparse storage of non-empty voxels: "Voxel-based detectors transform irregular point clouds into regular voxel-grids" (1. Introduction); "We define a dense voxel-grid, which has  $N_{dense}$  voxels in total, to rasterize the whole 3D scene. In practice we only maintain those non-empty voxels with a  $N_{sparse} \times 3$  integer indices array V" (3.2. Voxel Transformer Module)
- Fixed voxel size is assumed in experiments: "with a commonly-used 3D convolutional backbone [34] and the voxel size as (0.05m, 0.05m, 0.1m) on the KITTI dataset" (1. Introduction)
- Voxel center coordinates computed from integer indices and voxel size: "We first transform the indices  $v_i, v_j$  to the corresponding 3D coordinates of the real voxel centers  $p_i, p_j$  by  $p = r \cdot (v + 0.5)$ , where r is the voxel size." (3.2. Voxel Transformer Module)
- Downsampling changes voxel size: "Input voxels are downsampled 3 times with the stride 2 by 3 sparse voxel modules... the voxel size are doubled during downsampling." (A. Architecture)
- Initial feature projection from coordinates: "The input non-empty voxel coordinates are first transformed into 16-channel initial features by a linear projection layer" (Implementation Details, 4.1)
- Padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable length: The per-query attention neighborhood is fixed in experiments: "The number of total attending voxels is set to 48 for each querying voxel" (Implementation Details, 4.1); also constrained in design: "the number of attending voxels in  $\Omega(i)$  should be small enough, e.g. less than 50" (3.3. Efficient Attention Mechanism).
- Attention type: Local (windowed) and dilated sparse attention. Evidence: "we propose two attention mechanisms: Local Attention and Dilated Attention" (Abstract); "Local Attention focuses on the neighboring region" and "Dilated Attention obtains a large attention range with only a few attending voxels" (Abstract).
- Cost-control mechanisms: "With a carefully designed parameter list  $R_{dilated}$ , the attention range is able to reach more than 15m but the number of attending voxels for each querying voxel is still kept less than 50." (3.3. Efficient Attention Mechanism); "we propose Fast Voxel Query, which contains a GPU-based hash table to efficiently store and lookup the non-empty voxels." (Abstract)

## 8. Positional Encoding (Critical Section)

- Mechanism: Relative positional encoding using relative coordinates. Evidence: "the positional encoding  $E_{pos}$  can be calculated by:  $$E_{pos} = (p_i - p_j)W_{pos}. (2)$$" (3.2. Voxel Transformer Module); "self-attention on voxels is a natural 3D extension of standard 2D self-attention with sparse inputs and relative coordinates as positional embeddings." (3.2. Voxel Transformer Module)
- Where applied: Added to key and value projections in attention. Evidence: "$Q_i = f_i W_q, K_j = f_j W_k + E_{pos}, V_j = f_j W_v + E_{pos}$" (3.2. Voxel Transformer Module)
- Fixed across experiments / modified / ablated: Not specified.

## 9. Positional Encoding as a Variable

- Treatment: Not presented as a research variable; used as a fixed architectural component. Evidence: "self-attention on voxels is a natural 3D extension of standard 2D self-attention with sparse inputs and relative coordinates as positional embeddings." (3.2. Voxel Transformer Module)
- Multiple positional encodings compared: Not specified.
- PE claimed as not critical or secondary: Not specified.

## 10. Evidence of Constraint Masking

- Model sizes: "SECOND [34]... 5.3 M" vs "VoTr-SSD (ours)... 4.8M" and "PV-RCNN [26]... 13.1 M" vs "VoTr-TSD (ours)... 12.6M" (Table 7, 4.4. Ablation Studies).
- Dataset sizes: "The Waymo Open Dataset contains 1000 sequences in total, including 798 sequences (around 158k point cloud samples) in the training set and 202 sequences (around 40k point cloud samples) in the validation set." (4.1. Experimental Setup); "The KITTI dataset contains 7481 training samples and 7518 test samples" (4.1. Experimental Setup)
- Attribution of gains: "The significant performance gains in the far away area show the importance of large context information obtained by VoTr to 3D object detection." (4.2. Comparisons on the Waymo Open Dataset); "Dilated Attention guarantees larger receptive fields for each voxel and brings 2.79% moderate mAP gain compared to using only Local Attention." (4.4. Ablation Studies)
- Scaling data: Not emphasized; the paper notes a training subset on Waymo: "we uniformly sample 20% frames for training" (Training and Inference Details, 4.1)

## 11. Architectural Workarounds

- Sparse and submanifold voxel modules to handle sparsity and empty locations: "we propose the sparse voxel module and the submanifold voxel module, which can operate on the empty and non-empty voxel positions effectively" (Abstract); "submanifold voxel modules strictly operate on the non-empty voxels" and "sparse voxel modules can extract voxel features at the empty locations" (3.2. Voxel Transformer Module)
- Local and Dilated Attention to control attention range and cost: "Local Attention focuses on the neighboring region" and "Dilated Attention obtains a large attention range with only a few attending voxels" (Abstract); "the number of attending voxels in  $\Omega(i)$  should be small enough, e.g. less than 50, to avoid heavy computational overhead." (3.3. Efficient Attention Mechanism)
- Fast Voxel Query to accelerate sparse attention lookup: "we propose Fast Voxel Query, which contains a GPU-based hash table to efficiently store and lookup the non-empty voxels." (Abstract)
- Hierarchical downsampling with stride 2: "Input voxels are downsampled 3 times with the stride 2 by 3 sparse voxel modules." (A. Architecture)

## 12. Explicit Limitations and Non-Claims

- Limitation of naive Transformer on voxels: "directly applying standard Transformer modules to voxels is infeasible" (1. Introduction)
- Future work: "For future work, we plan to explore more Transformer-based architectures on 3D detection." (5. Conclusion)
- Explicit non-claims about open-world/multi-task learning: Not specified.

### 13. Constraint Profile (Synthesis)

- **Constraint Profile:** Domain scope: LiDAR point clouds in autonomous-driving datasets (Waymo/KITTI); no cross-modality evaluation is described.
- **Constraint Profile:** Task structure: 3D object detection only (vehicle/car detection) across datasets.
- **Constraint Profile:** Representation rigidity: voxel-grid rasterization with fixed voxel size and stride-2 downsampling; non-empty voxel indices drive attention.
- **Constraint Profile:** Model sharing vs specialization: separate detector variants (VoTr-SSD/VoTr-TSD) trained per dataset; no joint multi-task training described.
- **Constraint Profile:** Role of positional encoding: relative-coordinate positional encoding baked into attention, not varied or ablated.

### 14. Final Classification

**Single-task, single-domain.** The evaluation is limited to 3D object detection from point clouds ("3D object detection from point clouds" in the Abstract) on autonomous-driving LiDAR datasets ("Waymo Open dataset" and "KITTI" in 4. Experiments). The paper does not describe multi-task training or cross-domain transfer; training is described separately per dataset (Training and Inference Details, 4.1).
