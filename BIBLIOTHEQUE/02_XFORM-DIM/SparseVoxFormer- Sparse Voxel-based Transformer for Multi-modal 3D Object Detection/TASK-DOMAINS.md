# SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection (Not specified in the paper.)
Source: SparseVoxFormer- Sparse Voxel-based Transformer for Multi-modal 3D Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | LiDAR point-cloud sweeps and multi-view camera images | 2D (x, y); 3D (x, y, z) | Capped | Dynamic (inferred) | Constructed (inferred) | 3D bounding cuboids with class and motion attributes | 3D (x, y, z) | Capped |

## Summary
The paper covers one task: multi-modal 3D object detection for autonomous driving with LiDAR and camera inputs. The input space spans 2D image coordinates and 3D spatial LiDAR/voxel coordinates, while outputs are 3D cuboids in world space. The model interface is bounded by fixed query and token limits, so input and output dynamics are Capped. Attention is Dynamic (inferred) because Top-K features are selected at runtime, and state is Constructed (inferred) because the system builds sparse voxel and fused feature representations before prediction.

## Evidence
### Task: 3D object detection
- "3D object detection is a critical task in real-world applications such as autonomous driving." (Section 1. Introduction)
- "we target multi-modal 3D object detection and thus specifically focus on the nuScenes dataset, which is unique in that it is the only one to provide 360° view coverage and full multi-modality with LiDAR and camera sensors." (Section 2. Related Work)
- "since our 3D voxel features carry their 3D positional coordinates (x, y, z), they can be accurately projected to image feature space by:"
  (Section 3.2. Explicit Multi-modal Fusion with Sparse Features)
- "retaining the Top-K features based on the confidence score of the trained head, implying that the detector uses a fixed number of tokens." (Section 3.4. Redundant Feature Elimination)
- "The number of queries used in our model is 900." (Section A.2. Additional Architecture Detail)
- "the heads predict the center, scale, rotation, velocity, and class of each bounding cuboid." (Section A.2. Additional Architecture Detail)
- Inference: Attention Dynamic is marked Dynamic (inferred) because runtime Top-K confidence filtering selects which sparse tokens are passed forward (Section 3.4). State Dynamic is marked Constructed (inferred) because the model explicitly constructs sparse voxel states and fused multi-modal feature states ("the sparse features can be obtained by omitting zero-filled features and serializing valid feature cells" in Section 3; "F_{combined}^{sparse} = Concat(F_{lidar}^{sparse}, F_{image}^{(u,v)})" in Section 3.2).
