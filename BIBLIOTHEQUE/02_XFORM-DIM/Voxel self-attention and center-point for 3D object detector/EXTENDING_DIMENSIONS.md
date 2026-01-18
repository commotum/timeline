## 1. Basic Metadata

- Title: "Voxel self-attention and center-point for 3D object detector" (Article)
- Authors: "Likang Fan, Jie Cao, Xulei Liu, Xianyong Li, Liting Deng, Hongwei Sun, Yiqiang Peng" (Article)
- Year: "September 20, 2024" (Highlights)
- Venue (conference/journal/arXiv): "iScience 27, 110759" (Highlights)

## 2. One-Sentence Contribution Summary

The paper's primary contribution is stated as: "Therefore, in this paper, we propose the voxel self-attention and center-point (VSAC)." and it adds that "Firstly, a voxel self-attention network is designed into VSAC to capture extensive voxel relationship" and "Finally, we employ a center-point detection head to make the prediction direction closer to the real object during steering" (SUMMARY).

## 3. Tasks Evaluated

Task name: 3D object detection (Car-3D/Car-BEV/AOS)
Task type: Detection
Dataset(s) used: KITTI dataset
Domain: LiDAR point clouds (autonomous driving)
Quotes: "The KITTI dataset is extensively utilized for assessing 3D object detection" (Dataset and evaluation); "the evaluation metrics by the KITTI dataset, which include the mean average precision with 11 recall positions (AP|R11), the mean average precision with 40 recall positions (AP|R40) and the average orientation similarity (AOS)" (Dataset and evaluation); "In BEV detection, VSAC also has an increase" (Comparisons on the KITTI dataset)

Task name: 3D object detection
Task type: Detection
Dataset(s) used: Waymo Open Dataset
Domain: LiDAR point clouds (autonomous driving)
Quotes: "The Waymo Open Dataset has a total of 798 training set sequences with 158,361 LiDAR samples and 202 validation set sequences with 40,077 LiDAR samples" (Dataset and evaluation); "We used Waymo Open Dataset's officially recognized mAP and mAPH with rotated IoU threshold 0.7 for cars as indicators of performance" (Dataset and evaluation)

Task name: 3D object detection (10 categories)
Task type: Detection
Dataset(s) used: nuScenes dataset
Domain: LiDAR point clouds (autonomous driving)
Quotes: "The nuScenes dataset includes ten categories, with a total of 40,000 annotated frames" (Dataset and evaluation); "we evaluated the detection performance of VSAC on the nuScenes dataset and compared it with current state-of-the-art methods across ten detection categories" (Comparisons on the nuScenes dataset)

Task name: Online vehicle detection in campus scenes
Task type: Detection
Dataset(s) used: Campus LiDAR data collected via ROS (real-vehicle experiment)
Domain: Real-world LiDAR point clouds (campus autonomous driving scenes)
Quotes: "the experiments with the VSAC algorithm primarily utilized LiDAR to collect campus scene data through ROS and implemented online detection on the IPC" (Real vehicle detection experiment); "Lastly we conducted autonomous vehicle detection experiments on campus" (RESULTS)

## 4. Domain and Modality Scope

- Evaluation performed on a single domain with multiple datasets in the same modality: the paper focuses on "3D LiDAR object detection" (INTRODUCTION) and reports results on "KITTI dataset, Waymo Open Dataset, and nuScenes dataset" (SUMMARY). The real-vehicle experiment also uses LiDAR: "utilized LiDAR to collect campus scene data" (Real vehicle detection experiment).
- Multiple domains within the same modality? Not claimed; all cited evaluations are autonomous-driving LiDAR datasets and a campus driving experiment (SUMMARY; Real vehicle detection experiment).
- Multiple modalities? Not stated for VSAC evaluation; the task is framed as LiDAR-based ("3D LiDAR object detection") (INTRODUCTION).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D object detection (KITTI) | Not specified. | Not specified. | Not specified. | "VSAC was trained end-to-end on 2 NVIDIA-A30 GPUs using the ADAM optimizer. For the KITTI dataset, the batch size was set to 4" (Implementation details) |
| 3D object detection (Waymo Open Dataset) | Not specified. | Not specified. | Not specified. | "For the Waymo Open Dataset, the batch size was set to 2" (Implementation details) |
| 3D object detection (nuScenes) | Not specified. | Not specified. | Not specified. | "For the nuScenes dataset, the batch size was set to 2" (Implementation details) |
| Online vehicle detection in campus scenes | Not specified. | Not specified. | Not specified. | "implemented online detection on the IPC" (Real vehicle detection experiment) |

## 6. Input and Representation Constraints

- Voxelization with fixed quantization steps: "we first employ a mapping algorithm<sup>10,12,40</sup> to represent the original point clouds as voxels" and "[ V_{long} , V_{width} , V_{height} ] represent the quantization step size in voxelization" (Voxelization).
- Points-per-voxel handling: "if the number of points in a voxel exceeds a preset threshold, memory consumption is reduced by randomly discarding the excess points. Conversely, if there are not enough points in the voxel, we use the average value to fill in the points" (Voxelization).
- Voxel feature representation: "the average features of all points within a voxel are taken to represent the features of that voxel" (Voxelization).
- Fixed coordinate ranges and voxel sizes per dataset: "For the KITTI dataset, the x, y, z coordinate range of the point cloud is [0, +70.4], [-40, +40], [-3.0, +1.0] and the voxel size is set to [0.05, 0.05, 0.1]. For the Waymo Open Dataset, the x, y, z coordinate range of the point cloud is set to [-75.2, +75.2], [-75.2, +75.2], [-2.0, +4.0] and the voxel size is set to [0.1, 0.1, 0.15]. For nuScenes dataset, the x, y, z coordinate range of the point cloud is set to [-54.0, +54.0], [-54.0, +54.0], [-5.0, +3.0] and the voxel size is set to [0.075, 0.075, 0.2]" (Implementation details).
- Fixed feature dimensionality and detection capacity: "non-empty voxels are linearly projected to yield 16-channel initial features" and "the feature map scale factor is set at 1/4, with a maximum detection capacity of 100 objects" (Implementation details).
- Output stride downsampling: "Through downsampling by the output stride, we obtain the output prediction Y_1" and "Where Q represents the output stride" (Center-point detection head module).
- Feature-map scaling in experiments: "scaling factors of 1/4 and 1/8 are employed to alter the feature maps" (Effects of feature map scaling factors).

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; attention is bounded by the voxel neighborhood size, e.g., "The number of voxels in the attention range are set to 16, 32, and 48" (Effects of the number of voxels in attention calculation).
- Fixed or variable sequence length: Not explicitly stated; experiments use fixed attention-range sizes ("16, 32, and 48") (Effects of the number of voxels in attention calculation).
- Attention type: Sparse/windowed attention over voxels, with non-empty voxel focus: "the ripple-spread center-emanating attention range ( $\Omega(i)$ ) is proposed" (3D backbone module) and "attention-1 preserves the original structural information of the 3D space by solely calculating features of non-empty voxels. Meanwhile, the attention-2 enhances flexibility by additionally extracting features from non-empty voxel and a select few empty voxels" (Voxel self-attention).
- Computational cost controls: "self-attention calculations on non-empty voxels present challenges in real-time applications" (Methodology); "we propose the ripple-like center-spreading attention module to determine the attention range" (Methodology); "we establish an voxel hash table for storing non-empty voxels to expedite the voxel search process in self-attention computations" (Methodology); "each operation on the non-empty voxel is assigned to a separate CUDA thread" (Methodology); "a downsampling process is applied to the voxels" (Effects of downsampling voxel numbers).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Relative positional encoding from voxel-center differences: "The computation of the positional encoding E_p is as follows: E_p = (o_i - o_k)W_p" (Voxel self-attention).
- Where applied: Added to key and value embeddings inside attention: "K_k = f_k W_k + E_p" and "V_k = f_k W_v + E_p" (Voxel self-attention).
- Fixed/modified/ablated: Not specified; no alternative positional encodings or ablations are described.

## 9. Positional Encoding as a Variable

- The paper treats positional encoding as a fixed architectural component in the attention computation: "K_k = f_k W_k + E_p" and "E_p = (o_i - o_k)W_p" (Voxel self-attention).
- Multiple positional encodings compared: Not stated.
- Claims that PE choice is not critical: Not stated.

## 10. Evidence of Constraint Masking

- Model size(s): Model size not specified.
- Dataset size(s): "The KITTI dataset is extensively utilized for assessing 3D object detection and consists of 7,481 training samples and 7,518 testing samples" (Dataset and evaluation); "The Waymo Open Dataset has a total of 798 training set sequences with 158,361 LiDAR samples and 202 validation set sequences with 40,077 LiDAR samples" (Dataset and evaluation); "The nuScenes dataset contains 1,000 autonomous driving scenes, divided into a training set of 700 scenes, a validation set of 150 scenes, and a test set of 150 scenes" (Dataset and evaluation).
- Performance gains attributed to architectural components rather than scaling data/model size: "We propose the VSAC network framework for 3D LiDAR object detection, which utilizes voxel self-attention to learn wide-range relational between voxels" (INTRODUCTION); "In order to enhance the feature representation capability, we propose the PST-FPN to emphasize the importance of key features from various feature channels and resolutions by incorporating channel and spatial attention" (INTRODUCTION); "We propose a center-point detection head, which represents objects as points, thereby enabling more precise prediction of cars orientation throughout turning maneuvers" (INTRODUCTION); "AP|R40 improves by 1.35% when our algorithm adds the PST-FPN" (Effects of pseudo spatio-temporal feature pyramid net).
- Scaling attention/voxel counts affects results: "The number of voxels in the attention range are set to 16, 32, and 48" (Effects of the number of voxels in attention calculation); "When the number of voxels involved in attention is reduced from 48 to 16, the AP|R40 for Car decreased by 3.73%" (Effects of the number of voxels in attention calculation); "increasing the number of retained voxels from 9,000 to 18,000 during downsampling improved the AP|R40 for Car by 0.65%" (Effects of downsampling voxel numbers).
- Training tricks: "we employed a data augmentation strategy for the 3D point cloud data, which included random flipping, scaling within a range of 0.95-1.05, and rotation around the X axis between -5 and  $5^{\circ}$" (Dataset and data processing).

## 11. Architectural Workarounds

- Restricted attention range to manage cost: "we propose the ripple-like center-spreading attention module to determine the attention range" (Methodology).
- Fast voxel lookup: "we establish an voxel hash table for storing non-empty voxels to expedite the voxel search process in self-attention computations" (Methodology).
- Sparse voxel attention: "attention-1 preserves the original structural information of the 3D space by solely calculating features of non-empty voxels" (Voxel self-attention).
- Voxel downsampling: "a downsampling process is applied to the voxels" (Effects of downsampling voxel numbers).
- PST-FPN enhancements: "PST-FPN add two additional components: channel attention and spatial attention with residual networks to highlight different key features" (Pseudo spatio-temporal feature pyramid Net).
- Center-point detection head: "we employ a center-point detection head, which is effective regressing the 3D dimensions, orientation and other attribute information of the object through the center-point of the object" (Methodology).

## 12. Explicit Limitations and Non-Claims

- Limitations: "when the detected objects are too far away or occluded (resulting in sparse point cloud reflections), VSAC may fail to detect these objects" (Limitations of the study).
- Future work implied by limitations: "Future work aims to address these issues, ensuring that objects at a distance or under occlusion can still be accurately recognized" (Limitations of the study).
- Explicit non-claims about open-world, unrestrained multi-task, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: LiDAR-only autonomous-driving detection across KITTI, Waymo, nuScenes, plus campus LiDAR tests.
> – Task structure: Single core task of 3D object detection with BEV/AOS evaluation metrics.
> – Representation rigidity: Fixed voxel grid ranges and voxel sizes per dataset; fixed feature-map scale factor in the head.
> – Model sharing vs specialization: Training is described per dataset; weight sharing or joint multi-task training is not specified.
> – Role of positional encoding: Relative voxel-center encoding used in attention; treated as a fixed architectural component.

## 14. Final Classification

**Single-task, single-domain.** The paper focuses on "3D LiDAR object detection" (INTRODUCTION) and evaluates that single task across multiple autonomous-driving LiDAR datasets ("KITTI dataset, Waymo Open Dataset, and nuScenes dataset") (SUMMARY). Despite multiple datasets and a campus deployment, all evaluations remain within the LiDAR-based autonomous-driving detection domain, with no evidence of multi-task or multi-domain learning claims.
