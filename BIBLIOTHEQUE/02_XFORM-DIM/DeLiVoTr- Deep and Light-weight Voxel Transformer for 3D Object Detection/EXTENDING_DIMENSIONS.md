## 1. Basic Metadata
- Title: "DeLiVoTr: Deep and light-weight voxel transformer for 3D object detection" (Front matter)
- Authors: "Gopi Krishna Erabati\*, Helder Araujo" (Front matter)
- Year: "Received 23 January 2024; Received in revised form 3 March 2024; Accepted 13 March 2024" (Front matter)
- Venue: "Intelligent Systems with Applications" (Front matter)

## 2. One-Sentence Contribution Summary
It proposes a "Deep and Light-weight Voxel Transformer (DeLiVoTr) network with voxel intra- and inter-region transformer modules" to maintain same-scale feature maps and receptive field for LiDAR 3D object detection, targeting small objects in autonomous driving (Abstract).

## 3. Tasks Evaluated
### Task 1: 3D object detection (LiDAR point clouds)
- Task type: Detection
- Dataset(s): Waymo Open Dataset (WOD); KITTI dataset
- Domain: LiDAR point clouds in autonomous driving
- Evidence: "3D object detection is one of the fundamental tasks in autonomous driving and robotics." (Section 1. Introduction)
- Evidence: "The DeLiVoTr inputs LiDAR point cloud and predicts 3D bounding boxes in an autonomous driving scenario." (Section 3.1. Overview)
- Evidence: "We evaluate our DeLiVoTr model on large-scale publicly available autonomous driving datasets: Waymo Open Dataset (Sun et al., 2020) and KITTI dataset (Geiger et al., 2012)." (Section 4. Experiments)
- Evidence: "*vehicles*, *pedestrians* and *cyclist* categories are used for the evaluation." (Section 4.1. Implementation details - Waymo dataset)
- Evidence: "Small size objects like *pedestrian* and *cyclist* are used for the evaluation." (Section 4.1. Implementation details - KITTI dataset)

## 4. Domain and Modality Scope
- Single domain: Yes. "We evaluate our DeLiVoTr model on large-scale publicly available autonomous driving datasets: Waymo Open Dataset (Sun et al., 2020) and KITTI dataset (Geiger et al., 2012)." (Section 4. Experiments)
- Multiple domains within the same modality: No; both datasets are autonomous driving LiDAR. "The DeLiVoTr inputs LiDAR point cloud and predicts 3D bounding boxes in an autonomous driving scenario." (Section 3.1. Overview)
- Multiple modalities: No; evaluation uses LiDAR point clouds. "Each sample consists of a LiDAR point cloud captured by a 64-beam LiDAR sensor." (Section 4.1. Implementation details - KITTI dataset)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D object detection | Not specified (single task; separate training schedules per dataset are described) | Not specified | Not specified (single detection head described) | "We train the model for 24 and 40 epochs with a batch size of 2 and 4 for Waymo and KITTI datasets respectively" (Section 4.1. Implementation details); "For the detection head in the decoder we employ the CenterPoint (Yin et al., 2021) head" (Section 3.4. Decoder) |

## 6. Input and Representation Constraints
- 3D point cloud input: "Given a point cloud  $\mathcal{P} = \{p_i\}_{i=1}^N$ , where  $p_i \in \mathbb{R}^3$" (Section 3.2. Voxelization).
- Fixed voxel size/resolution: "the voxel size  $(v_x, v_y, v_z)$" (Section 3.2. Voxelization); "The input data processing such as point cloud range, voxel resolution of LiDAR point clouds for different datasets is given in Table 1." (Section 4.1. Implementation details)
- Fixed grid shape: "The point cloud is discretized into a sparse grid of shape  $(s_x, s_y, s_z)$ ." (Section 3.2. Voxelization)
- Fixed point cloud range/extent (Waymo example): "Each point cloud covers a scene of 150 m  $\times$  150 m area." (Section 4.1. Implementation details - Waymo dataset)
- Region partitioning: "Instead, we divide the voxels ( $\mathcal{V}$ ) into non-overlapping regions with size ( $r_x, r_y, r_z$ ) similar to (Fan et al., 2022), so that the voxels within a region interact with each other to maintain the required receptive field." (Section 3.3.2. Voxel intra-region transformer)
- Variable number of voxels/tokens: "dynamic voxelization (Zhou et al., 2020) which completely maps the points to their corresponding voxels with dynamic number of voxels and points." (Section 3.2. Voxelization)
- BEV rasterization with zero-filled empty cells: "The sparse semantically rich voxel features from the encoder are rasterized to a BEV feature map according to their respective spatial locations. The spatial locations with no voxels are filled with zeros." (Section 3.4. Decoder)

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified; a typical per-region sequence length is noted as "~150" voxels. "lesser number of voxels (~150) (sequence length) in each local region" (Section 4.2.2. Ablation studies)
- Fixed or variable sequence length: Variable. "As the LiDAR point clouds are sparse in nature the number of voxels in each region varies." (Section 3.3.2. Voxel intra-region transformer)
- Attention type: Windowed/local intra-region attention plus inter-region attention (hierarchical/global at region level). "Instead, we divide the voxels ( $\mathcal{V}$ ) into non-overlapping regions with size ( $r_x, r_y, r_z$ ) similar to (Fan et al., 2022), so that the voxels within a region interact with each other to maintain the required receptive field." (Section 3.3.2. Voxel intra-region transformer); "we aggregate the features in each region and model the interactions between the regions" (Section 1. Introduction)
- Computational cost mechanisms: Region aggregation and lightweight attention. "The voxel feature region aggregation not only helps to reduce the computational complexity" (Section 1. Introduction); "we can replace the MHSA with single-head self-attention (SHSA) and FFN with lightweight FFN, reducing total number of parameters." (Section 3.3.1. DeLiVoTr block)

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Not specified beyond "PE is Positional Encoding (Carion et al., 2020)" (Section 3.3.2. Voxel intra-region transformer)
- Where applied: In attention inputs for intra- and inter-region transformers. "\text{SHSA}(\mathcal{G}(\mathcal{F}_{r}, \text{PE}(\mathcal{C}_{r})))" (Section 3.3.2. Voxel intra-region transformer); "\mathbf{SHSA}(\mathcal{G}(\mathcal{F}, \mathbf{PE}(\mathcal{C})))" (Section 3.3.3. Voxel inter-region transformer)
- Fixed/modified/ablated: Not specified; no ablation or alternative PE comparison is reported in the provided text.

## 9. Positional Encoding as a Variable
- Core research variable vs fixed assumption: Fixed architectural component; only defined as part of the attention equations. "PE is Positional Encoding (Carion et al., 2020)" (Section 3.3.2. Voxel intra-region transformer)
- Multiple positional encodings compared: Not stated.
- Claims PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking
- Model sizes: "The DeLiVoTr\_small model efficiently allocates the parameters among the different encoder layers with 1.56M parameters (occupy 3.7 GB of GPU memory), not only achieves improved performance in terms of LEVEL\_1 AP compared to SST but also achieves 20.5 FPS inference speed (20% more than SST)." (Section 4.2.1. Variants of DeLiVoTr)
- Dataset sizes: "The Waymo Open Dataset (Sun et al., 2020) consists of 798, 202 and 150 scenes for training, validation and testing respectively. Each consists of more than 200 K LiDAR point clouds" (Section 4.1. Implementation details - Waymo dataset); "The KITTI dataset (Geiger et al., 2012) consists of 3,712 and 3,769 samples for training and validation." (Section 4.1. Implementation details - KITTI dataset)
- Performance gains attributed to architectural hierarchy: "As our network maintains the same scale of voxel feature maps without any downsampling, there is no semantic information loss which also helps in the increase of performance of small objects, such as *pedestrians*." (Section 4.2.1. Results and discussion)
- Scaling model size: "scaling in the DeLiVoTr block learns the wider and deeper voxel feature representation which improves the performance as shown in Table 9." (Section 4.2.2. Ablation studies)
- Training tricks: "We apply fade strategy by disabling copy-and-paste augmentation (Yan et al., 2018) for the last quarter epochs." (Section 4.1. Implementation details)
- Scaling data: Not claimed.

## 11. Architectural Workarounds
- Windowed attention to reduce complexity: "Instead, we divide the voxels ( $\mathcal{V}$ ) into non-overlapping regions with size ( $r_x, r_y, r_z$ ) similar to (Fan et al., 2022), so that the voxels within a region interact with each other to maintain the required receptive field." (Section 3.3.2. Voxel intra-region transformer)
- Region-level aggregation and inter-region attention for larger receptive field at lower cost: "we aggregate the features in each region and model the interactions between the regions to increase the receptive field size. The voxel feature region aggregation not only helps to reduce the computational complexity" (Section 1. Introduction)
- Lightweight attention/FFN: "we can replace the MHSA with single-head self-attention (SHSA) and FFN with lightweight FFN, reducing total number of parameters." (Section 3.3.1. DeLiVoTr block)
- Layer-level scaling (variable-sized encoder layers): "we introduce layer-level scaling of DeLiVoTr blocks in the encoder layers that allows variable-sized layers instead of uniform stacking of encoder layers" (Section 3.3.1. DeLiVoTr block)
- Single-scale feature maps (no downsampling): "we adopt the single-scale feature map design" (Section 3.1. Overview)

## 12. Explicit Limitations and Non-Claims
No explicit limitations, future work, or non-claims are stated in the provided OCR text.

## 13. Constraint Profile (Synthesis)
**Constraint Profile:**
- Domain scope: Single autonomous driving LiDAR domain (Waymo and KITTI only).
- Task structure: Single task (3D object detection of vehicles/pedestrians/cyclists).
- Representation rigidity: Fixed point-cloud range/voxel resolution settings (reported in Table 1); fixed region partitioning; BEV rasterization with zero-filled empty cells; dynamic voxelization yields variable token counts.
- Model sharing vs specialization: No multi-task sharing described; training schedules are specified separately for Waymo vs KITTI.
- Role of positional encoding: Used as a fixed component in intra- and inter-region attention; not compared or ablated.

## 14. Final Classification
**Single-task, single-domain.** The paper frames its problem as "3D object detection" and the model "inputs LiDAR point cloud and predicts 3D bounding boxes in an autonomous driving scenario" (Sections 1 and 3.1). Evaluation is limited to autonomous-driving LiDAR datasets - "Waymo Open Dataset" and "KITTI dataset" - with no cross-domain or multi-task claims (Section 4).
