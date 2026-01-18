## 1. Basic Metadata

Title: "Point-voxel dual transformer for LiDAR 3D object detection*" (Front matter)

Authors: "TONG Jigang<sup>1</sup>, YANG Fanhang<sup>1</sup>, YANG Sen<sup>1</sup>, and DU Shengzhi<sup>2</sup>**" (Front matter)

Year: "(Received 17 July 2023; Revised 2 March 2025) ©Tianjin University of Technology 2025" (Front matter)

Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper presents "a two-stage light detection and ranging (LiDAR) three-dimensional (3D) object detection framework ... namely point-voxel dual transformer (PV-DT3D)" to improve LiDAR 3D object detection from point clouds. (Front matter)

## 3. Tasks Evaluated

Task 1

Task name: LiDAR 3D object detection (car category on KITTI).

Task type: Detection.

Dataset(s) used: KITTI dataset.

Domain: LiDAR point clouds for autonomous driving.

Quotes: "Three-dimensional (3D) object detection from point clouds for autonomous driving attracts increasing interest in the field of deep learning." (Section 1. Introduction) "The KITTI dataset is utilized for subsequent experiments, which includes 7 481 training samples and 7 518 test samples." (Section 4.1 KITTI dataset) "The commonly used \"car\" category of KITTI dataset is used for experiments." (Section 4.3 Detection performance of the KITTI dataset)

## 4. Domain and Modality Scope

Single domain: Yes. Evidence: "The KITTI dataset is utilized for subsequent experiments" and the task is framed as "object detection from point clouds for autonomous driving." (Section 4.1 KITTI dataset; Section 1. Introduction)

Multiple domains within the same modality: Not indicated.

Multiple modalities: Not indicated; the paper specifies "light detection and ranging (LiDAR) 3D object detection." (Front matter)

Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| LiDAR 3D object detection (KITTI car) | Yes (single task; end-to-end training described) | Not specified | Yes (separate FFNs for confidence and box refinement) | "The proposed PV-DT3D is trained end-to-end against the first-stage proposal generation loss ... and the second-stage refinement loss" (Section 3.5 Training losses); "The global proposal representation is fed into two separate FFNs for confidence prediction and bounding box refinement, respectively." (Section 3.4 Detect head and training objectives) |

## 6. Input and Representation Constraints

- Fixed raw point sampling: "Firstly, 3 072 raw points are randomly sampled by FPS." (Section 4.2.2 Training and inference details)
- Fixed internal keypoints: "Then in the dual transformer, 256 internal keypoints are randomly selected for subsequent processing." (Section 4.2.2 Training and inference details)
- Padding to fixed length: "If the number of internal keypoints is less than 256, dummy points are padded to ensure 256 points for achieving parallel running of the dual transformer." (Section 4.2.2 Training and inference details)
- Input includes coordinates and reflectance: "given an N-points 3D scene with position coordinates and reflectance" (Section 3.1 3D proposal generation and keypoints sampling)
- Voxelization step: "The raw points are firstly voxelized in the form of region proposal networks (RPN) for high quality proposals." (Section 3. Methodology)

## 7. Context Window and Attention Structure

Maximum sequence length: 256 internal keypoints for the dual transformer; "256 internal keypoints are randomly selected for subsequent processing" and padded to 256 if fewer. (Section 4.2.2 Training and inference details)

Fixed or variable sequence length: Fixed to 256 via padding; "dummy points are padded to ensure 256 points for achieving parallel running of the dual transformer." (Section 4.2.2 Training and inference details)

Attention type: Global point-wise and channel-wise attention in a dual transformer; "the dual transformer encoder-decoder architecture is proposed for bounding-box refinement, taking advantages of both point-wise and channel-wise transformers." (Section 3.3.2 Dual transformer for proposal refinement) The point-wise branch is "the point-wise multi-head cosh-attention encoder-decoder architecture." (Section 3.3.2)

Mechanisms to manage computational cost: Cosh-attention replaces vanilla attention for lower complexity; "The cosh-attention is used to replace vanilla attention for lower spatial and temporal complexity." (Section 3.3.2 Dual transformer for proposal refinement)

## 8. Positional Encoding (Critical Section)

Positional encoding mechanism: Implicit / none; "the position embedding is not used in the dual transformer because the features already contain spatial position information." (Section 3.3.2 Dual transformer for proposal refinement)

Where it is applied: Not applied; "the position embedding is not used in the dual transformer." (Section 3.3.2 Dual transformer for proposal refinement)

Fixed across experiments vs modified or ablated: Fixed (not used); no alternatives or ablations mentioned. Evidence: "the position embedding is not used in the dual transformer" (Section 3.3.2 Dual transformer for proposal refinement)

## 9. Positional Encoding as a Variable

The paper treats positional encoding as a fixed architectural assumption (not used): "the position embedding is not used in the dual transformer because the features already contain spatial position information." (Section 3.3.2 Dual transformer for proposal refinement)

Multiple positional encodings compared: Not stated.

Claims that PE choice is not critical or secondary: Not stated beyond the rationale for omitting it.

## 10. Evidence of Constraint Masking

Model size(s): Model size not specified.

Dataset size(s): "The KITTI dataset is utilized for subsequent experiments, which includes 7 481 training samples and 7 518 test samples." (Section 4.1 KITTI dataset)

Primary attribution of gains: Architectural modules and fusion strategies, not scaling data or model size. Evidence: "Ablation evaluations confirm the effectiveness of the proposed proposal-aware VSA module and dual transformer for 3D object detection." (Section 1. Introduction) "A series of ablation studies are conducted for verifying the effectiveness of the point-voxel fusion features, proposal-aware VSA module, and the proposed dual transformer" (Section 4.4 Ablation studies)

## 11. Architectural Workarounds

- Two-stage detection with proposals and refinement: "a two-stage light detection and ranging (LiDAR) three-dimensional (3D) object detection framework" (Front matter) and "The raw points are firstly voxelized in the form of region proposal networks (RPN) for high quality proposals." (Section 3. Methodology)
- Keypoint sampling to reduce computation: "the furtherest point sampling (FPS) algorithm is adopted to select representative points" (Section 3. Methodology)
- Proposal-aware VSA to stabilize training: "we present an improved proposal-aware VSA module" and "enhances the local correlations among input points within the same proposal, thus stabilizes the training" (Section 3.2.2 Proposal-aware VSA module)
- Dual transformer with point-wise and channel-wise branches for refinement: "the dual transformer encoder-decoder architecture is proposed for bounding-box refinement, taking advantages of both point-wise and channel-wise transformers." (Section 3.3.2 Dual transformer for proposal refinement)
- Cosh-attention to reduce complexity: "The cosh-attention is used to replace vanilla attention for lower spatial and temporal complexity." (Section 3.3.2 Dual transformer for proposal refinement)
- Padding to fixed keypoint count for parallelism: "dummy points are padded to ensure 256 points for achieving parallel running of the dual transformer." (Section 4.2.2 Training and inference details)

## 12. Explicit Limitations and Non-Claims

Future work / limitations: "In our future research, we are committed to enhancing the accuracy of small target detection. We are actively exploring strategies, including point cloud completion and making modifications to the transformer architecture." (Section 5. Conclusion)

Explicit non-claims about multi-task or open-world learning: Not stated.

### 13. Constraint Profile (Synthesis)

- Domain scope: Single autonomous-driving LiDAR dataset (KITTI) and car-category evaluation.
- Task structure: Single 3D object detection task with two-stage proposal and refinement pipeline.
- Representation rigidity: Fixed sampling (3 072 raw points, 256 internal keypoints) with padding to 256 for the dual transformer.
- Model sharing vs specialization: Single end-to-end model with separate FFNs for confidence and box refinement; no multi-task sharing described.
- Role of positional encoding: Omitted as a fixed architectural assumption because spatial positions are already embedded in features.

### 14. Final Classification

Single-task, single-domain.

The evaluation is confined to a single LiDAR 3D object detection task on KITTI, explicitly using the "car" category and KITTI dataset. Evidence includes "The KITTI dataset is utilized for subsequent experiments" and "The commonly used \"car\" category of KITTI dataset is used for experiments." (Section 4.1 KITTI dataset; Section 4.3 Detection performance of the KITTI dataset)
