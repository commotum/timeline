## 1. Basic Metadata

- Title: "VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion" (Title header)
- Authors: "Yiming Li $^1$  Zhiding Yu $^{2*}$  Christopher Choy $^2$  Chaowei Xiao $^{2,3}$  Jose M. Alvarez $^2$  Sanja Fidler $^{2,4,5}$  Chen Feng $^1$  Anima Anandkumar $^{2,6}$" (Authors line)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes "VoxFormer, a Transformerbased semantic scene completion framework that can output complete 3D volumetric semantics from only 2D images" (Abstract).

## 3. Tasks Evaluated

**Task 1**
Task name: Scene completion (geometry/occupancy)
Task type: Reconstruction
Dataset(s) used: SemanticKITTI
Domain: Outdoor driving scenes; 3D voxel grids
Quotes: "We employ intersection over union (IoU) to evaluate the scene completion quality, regardless of the allocated semantic labels." (Section 4.1 Evaluation metrics) "We verify VoxFormer on SemanticKITTI [5], which provides dense semantic annotations for each LiDAR sweep from the KITTI Odometry Benchmark [71] composed of 22 outdoor driving scenarios." (Section 4.1 Dataset)

**Task 2**
Task name: Semantic segmentation (per-voxel semantics)
Task type: Segmentation
Dataset(s) used: SemanticKITTI
Domain: Outdoor driving scenes; 3D voxel grids
Quotes: "We use the mean IoU (mIoU) of 19 semantic classes to assess the performance of semantic segmentation." (Section 4.1 Evaluation metrics) "Output dense semantic map  $\mathbf{Y}_t \in \mathbb{R}^{H \times W \times Z \times (M+1)}$  by up-sampling and linear projection of  $\mathbf{F}_t^{3D}$ ." (Section 3.2 Overall Architecture)

## 4. Domain and Modality Scope

- Single domain (outdoor driving scenes): "SemanticKITTI [5], which provides dense semantic annotations for each LiDAR sweep from the KITTI Odometry Benchmark [71] composed of 22 outdoor driving scenarios." (Section 4.1 Dataset)
- Multiple domains within the same modality? Not indicated; only SemanticKITTI driving scenes are described.
- Multiple modalities? Camera images only: "We aim to predict a dense semantic scene within a certain volume in front of the vehicle, given only RGB images." (Section 3.1 Problem setup)
- Domain generalization or cross-domain transfer claimed? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Scene completion (geometry/occupancy) | Yes; same voxel grid output used for occupancy and semantics. | Not specified. | Not specified; single output includes empty vs semantic classes. | "use as output a voxel grid  $\mathbf{Y}_t \in \{c_0, c_1, ..., c_M\}^{H \times W \times Z}$  ... each voxel is either empty (denoted by  $c_0$ ) or occupied by a certain semantic class" (Section 3.1 Problem setup); "Output dense semantic map  $\mathbf{Y}_t \in \mathbb{R}^{H \times W \times Z \times (M+1)}$" (Section 3.2 Overall Architecture). |
| Semantic segmentation (per-voxel) | Yes; same voxel grid output used for all classes. | Not specified. | Not specified; single output includes M+1 classes. | "Output dense semantic map  $\mathbf{Y}_t \in \mathbb{R}^{H \times W \times Z \times (M+1)}$" (Section 3.2 Overall Architecture); "There is also a linear layer that projects feature dimension 128 to the number of classes 20." (Section 4.1 Implementation details). |

## 6. Input and Representation Constraints

- Input modality fixed to RGB images: "given only RGB images." (Section 3.1 Problem setup)
- Temporal input is allowed but defined as a set of images: "we use as input current and previous images denoted by  $\mathbf{I}_t = \{I_t, I_{t-1}, ...\}$" (Section 3.1 Problem setup); "our framework supports the input of single or multiple images." (Figure 2 caption)
- Output is a fixed voxel grid: "use as output a voxel grid  $\mathbf{Y}_t \in \{c_0, c_1, ..., c_M\}^{H \times W \times Z}$" (Section 3.1 Problem setup)
- Fixed scene volume and voxel size on SemanticKITTI: "SemanticKITTI SSC benchmark is interested in a volume of 51.2m ahead of the car, 25.6m to left and right side, and 6.4m in height. The voxelization of this volume leads to a group of 3D voxel grids with a dimension of  $256 \times 256 \times 32$  since each voxel has a size of  $0.2m \times 0.2m \times 0.2m$ ." (Section 4.1 Dataset)
- Fixed input image crop size in experiments: "we crop RGB images of cam2 to size  $1220 \times 370$" (Section 4.1 Implementation details)
- Fixed patch size: Not specified.
- Padding/resizing requirements beyond cropping: Not specified.
- Fixed number of voxel queries: "We pre-define a total of  $N_q$  voxel queries as a cluster of 3D-grid-shaped learnable parameters  $\mathbf{Q} \in \mathbb{R}^{h \times w \times z \times d}$   $(N_q = h \times w \times z)$" (Section 3.3 Voxel queries)
- Query grid resolution is lower than output resolution: "with  $h \times w \times z$  its spatial resolution which is lower than output resolution  $H \times W \times Z$  to save computations." (Section 3.3 Voxel queries)
- Depth-correction occupancy map uses a lower fixed resolution: "$\mathbf{M}_{out} \in \{0,1\}^{h \times w \times z}$  has a lower resolution than the input  $\mathbf{M}_{in} \in \{0,1\}^{H \times W \times Z}$" (Section 3.4 Depth correction)
- Fixed sizes in implementation: "using as input a voxelized pseudo point cloud with a size of  $256 \times 256 \times 32$  and as output an occupancy map with a size of  $128 \times 128 \times 16$ ." (Section 4.1 Implementation details)

## 7. Context Window and Attention Structure

- Maximum sequence length (voxel tokens): fixed to the predefined voxel queries; explicit size appears in ablation: "randomly proposing a subset from all  $128 \times 128 \times 16$  voxel queries" (Section 4.2.3 Query mechanism)
- Sequence length fixed or variable: fixed total queries, variable subset for cross-attention: "Generate class-agnostic query proposals  $\mathbf{Q}_p \in \mathbb{R}^{N_p \times d}$  which is a subset of the predefined voxel queries  $\mathbf{Q} \in \mathbb{R}^{N_q \times d}$" (Section 3.2 Overall Architecture)
- Attention type: deformable (sparse/local) cross-attention and self-attention: "For efficiency, we utilize deformable attention [66], which interacts with local regions of interest, and only sample  $N_s$  points around the reference point to compute the attention results." (Section 3.5 Stage-2) "Deformable cross-attention." (Section 3.5) "Deformable self-attention." (Section 3.5)
- Computational cost controls: "start from a sparse set of visible and occupied voxel queries from depth estimation" (Abstract); "using a sparse representation instead of a dense one is certainly more efficient and scalable." (Contributions); "save computations and memories by removing many empty spaces" (Section 3.4 Query proposal); "with  $h \times w \times z$  its spatial resolution which is lower than output resolution  $H \times W \times Z$  to save computations." (Section 3.3 Voxel queries)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Not specified beyond "learnable positional embeddings." "learnable positional embeddings will be added to voxel queries for attention stages" (Section 3.3 Voxel queries)
- Where it is applied: voxel queries and mask tokens at the input to attention: "learnable positional embeddings will be added to voxel queries for attention stages" (Section 3.3 Voxel queries); "The positional embeddings are also added to help mask tokens be aware of their 3D locations." (Section 3.3 Mask token)
- Fixed across experiments / modified per task / ablated: Not specified; no ablations or comparisons of positional encodings are described.

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption? Fixed architectural assumption only: "learnable positional embeddings will be added to voxel queries for attention stages" (Section 3.3 Voxel queries)
- Multiple positional encodings compared? Not specified.
- Claim that PE choice is not critical or secondary? Not specified.

## 10. Evidence of Constraint Masking

- Model size(s): "VoxFormer has a total of  $\sim$ 60M parameters, which is more lightweight than MonoScene with  $\sim$ 150M parameters." (Section 4.2.1 Our superiority in size and memory)
- Dataset size(s): "SemanticKITTI [5] ... composed of 22 outdoor driving scenarios." (Section 4.1 Dataset)
- Performance gains attributed to architecture, not scaling: "Such a large improvement stems from stage-1 with explicit depth estimation and correction" (Section 4.2.1 Comparison against camera-based methods)
- Resource constraints/memory: "VoxFormer needs less than 16GB GPU memory during training." (Section 4.2.1 Our superiority in size and memory)

## 11. Architectural Workarounds

- Two-stage sparse-to-dense design: "Our framework adopts a two-stage design where we start from a sparse set of visible and occupied voxel queries from depth estimation, followed by a densification stage that generates dense 3D voxels from the sparse ones." (Abstract)
- Masked autoencoder-style completion: "we apply a masked autoencoder design to propagate the information to all the voxels by self-attention." (Abstract)
- Sparsity as a scalability strategy: "since a large volume of the 3D space is usually unoccupied, using a sparse representation instead of a dense one is certainly more efficient and scalable." (Contributions)
- Deformable attention for local/sparse computation: "we utilize deformable attention [66], which interacts with local regions of interest, and only sample  $N_s$  points around the reference point" (Section 3.5 Stage-2)
- Query proposal to reduce computation: "save computations and memories by removing many empty spaces" (Section 3.4 Query proposal)

## 12. Explicit Limitations and Non-Claims

- Limitation / future work: "Our performance at long range still needs to be improved, because the depth is very unreliable at the corresponding locations. Decoupling the long-range and short-range SSC is a potential solution to enhance the SSC far away from the ego vehicle. We leave this as our future work." (Section 4.2.4 Limitation and future work)
- Explicit non-claims (open-world, unrestrained multi-task, meta-learning, etc.): Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single outdoor driving domain (SemanticKITTI) with camera-based inputs.
> – Task structure: One SSC pipeline evaluated via scene completion and semantic segmentation metrics.
> – Representation rigidity: Fixed voxel grids and fixed query grid sizes; fixed input crop size in experiments.
> – Model sharing vs specialization: Single output head over M+1 classes; geometry and semantics share weights.
> – Role of positional encoding: Learnable positional embeddings added to voxel queries/mask tokens; no comparison reported.

### 14. Final Classification

**Single-task, single-domain.** The paper evaluates camera-based semantic scene completion on a single dataset from one domain: "SemanticKITTI [5] ... composed of 22 outdoor driving scenarios" (Section 4.1 Dataset). The task is SSC with joint geometry and semantics: "semantic scene completion (SSC) [1] was proposed to jointly infer the complete scene geometry and semantics from limited observations." (Section 1 Introduction).
