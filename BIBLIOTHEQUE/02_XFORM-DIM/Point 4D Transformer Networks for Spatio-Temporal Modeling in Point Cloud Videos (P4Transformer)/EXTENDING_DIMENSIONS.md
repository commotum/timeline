## 1. Basic Metadata

- Title: "Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos" (Title page)
- Authors: "Hehe Fan" (Title page); "Yi Yang ReLER University of Technology Sydney" (Title page); "Mohan Kankanhalli School of Computing National University of Singapore" (Title page)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes "a novel Point 4D Transformer (P4Transformer) network to model raw point cloud videos" in order "to avoid point tracking" while capturing "the spatio-temporal structure in raw point cloud videos" (Abstract; 1. Introduction).

---

## 3. Tasks Evaluated

- Task name: 3D action recognition
  - Task type: Classification
  - Dataset(s) used: MSR-Action3D; NTU RGB+D 60; NTU RGB+D 120
  - Domain: Point cloud videos (3D point cloud video)
  - Quotes: "We evaluate our P4Transformer on a video-level classification task, *i.e.*, 3D action recognition" (1. Introduction). "Action recognition is a fundamental task for video modeling, which can be seen as a videolevel classification task" (3.3). "Experiments on the MSR-Action3D [28], NTU RGB+D 60 [45], NTU RGB+D 120 [30] and Synthia 4D [6] datasets" (1. Introduction). "given a point cloud video, we first use a point 4D convolution layer" (3.3).

- Task name: 4D semantic segmentation
  - Task type: Segmentation
  - Dataset(s) used: Synthia 4D
  - Domain: Point cloud videos (3D point cloud video)
  - Quotes: "a point-level prediction task, *i.e.*, 4D semantic segmentation" (1. Introduction). "The 4D semantic segmentation can be seen as a point-level classification task" (3.3). "Synthia 4D [6] uses the Synthia dataset [44] to create 3D videos" (4.2).

---

## 4. Domain and Modality Scope

- Domain scope: Single domain (point cloud videos). Evidence: "point cloud videos" (Abstract; 1. Introduction).
- Multiple domains within the same modality? Not specified.
- Multiple modalities? Not claimed; inputs are point cloud coordinates/features, with optional point colors: "Let  $P_t \in \mathbb{R}^{3 \times N}$  and  $F_t \in \mathbb{R}^{C \times N}$  denote the point coordinates and features of the t-th frame in a point cloud video" and "For Synthia 4D [6], the point colors are provided" (3.1).
- Domain generalization or cross-domain transfer? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D action recognition | Not specified. | Not specified. | Yes. | "Finally, an MLP layer converts the global feature to action predictions" (3.3). |
| 4D semantic segmentation | Not specified. | Not specified. | Yes. | "After the last feature interpolation layer, we add an MLP layer that converts point features to point predictions" (3.3). |

---

## 6. Input and Representation Constraints

- Fixed number of points per frame for action recognition: "we sample 2,048 points for each frame" (4.1).
- Fixed number of frames per clip (action recognition): "Point cloud videos are split into multiple clips (with a fixed number of frames) as inputs" (4.1).
- Fixed clip length for segmentation: "we conduct experiments on video clips with length of 3 frames" (4.2).
- Subsampling of frames and points: "we first select some frames based on the temporal stride  $s_t$ ." and "we use the farthest point sampling (FPS) [43] to subsample  $N' = N/s_s$  points" (3.1).
- Fixed dimensionality of spatial coordinates plus time: "Let  $P_t \in \mathbb{R}^{3 \times N}$  and  $F_t \in \mathbb{R}^{C \times N}$  denote the point coordinates and features of the t-th frame in a point cloud video" and "anchor coordinates, *i.e.*, (x, y, z, t)" (3.1; 3.2.1).
- Fixed/variable input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified (tokens not discussed); experiments specify fixed points per frame and fixed clip length ("we sample 2,048 points for each frame"; "fixed number of frames"; "length of 3 frames") (4.1; 4.2).
- Padding/resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; sequence length is described as L'N' after subsampling (" $I \in \mathbb{R}^{C' \times L'N'}$  is the self-attention input") (3.2.1).
- Sequence length fixed or variable: Fixed per experiment via clips, e.g., "fixed number of frames" for action recognition and "length of 3 frames" for segmentation (4.1; 4.2).
- Attention type: Global (video-level) self-attention: "the softmax function is performed on the entire video" and "we employ the video-level self-attention" (3.2.2).
- Computational cost mechanisms: "point 4D convolution reduces the number of points to be processed by the subsequent transformer" (1. Introduction); "we first select some frames based on the temporal stride  $s_t$ ." and "we use the farthest point sampling (FPS) [43] to subsample  $N' = N/s_s$  points" (3.1); "Third, a max pooling merges the transformed local features to a single global one" (3.3).

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Absolute 4D coordinate embedding added to features: "Therefore, we combine anchor coordinates, *i.e.*, (x, y, z, t), and local area features as the input to our transformer," and "where  $W_i \in \mathbb{R}^{C' \times 4}$  is the weight to convert 4D coordinates" (3.2.1).
- Where it is applied: Input embedding before self-attention (3.2.1).
- Fixed across all experiments / modified per task / ablated: Not specified.

---

## 9. Positional Encoding as a Variable

- Treatment: Fixed architectural assumption (coordinate embedding is part of the input) (3.2.1).
- Multiple positional encodings compared: Not specified.
- PE claimed "not critical" or secondary: Not specified.

---

## 10. Evidence of Constraint Masking

- Model size details: "The transformer contains 5 self-attention (m=5) blocks, with 8 heads (h=8) per block" (4.1). "the total feature dimension is fixed to 1024" in head-count ablation (4.3.2).
- Dataset sizes: MSR-Action3D: "The MSR-Action3D [28] dataset consists of 567 Kinect v1 depth videos, including 20 action categories and 23K frames in total" (4.1.1). NTU RGB+D 60: "It consists of 56K videos, with 60 action categories and 4M frames in total" (4.1.2). NTU RGB+D 120: "It consists of 114K videos, with 120 action categories and 8M frames in total" (4.1.2). Synthia 4D: "use the same training/validation/test split, with 19,888/815/1,886 frames, respectively" (4.2).
- Scaling model size effects: "with more transformer layers, P4Transformer can achieve better accuracy. However, too many layers decrease performance" and "using more heads can effectively increase accuracy. However, using too many heads makes the feature dimension of each head too short" (4.3.2).
- Primary attributed gains: Architectural choices emphasized in the contributions, including avoiding tracking and point 4D convolution: "to avoid point tracking, we propose a novel Point 4D Transformer" and "we propose a point 4D convolution" (1. Introduction).

---

## 11. Architectural Workarounds

- Point 4D convolution to reduce transformer load: "point 4D convolution reduces the number of points to be processed by the subsequent transformer" (1. Introduction).
- Frame and point subsampling: "we first select some frames based on the temporal stride  $s_t$ ." and "we use the farthest point sampling (FPS) [43] to subsample  $N' = N/s_s$  points" (3.1).
- Hierarchical reduction for segmentation: "we stack multiple point 4D convolution layers to exponentially reduce the number of points to be processed by the transformer" (3.3).
- Feature propagation to recover dense outputs: "we add feature propagation layers to interpolate point features" (3.3).
- Global pooling head for classification: "Third, a max pooling merges the transformed local features to a single global one" (3.3).

---

## 12. Explicit Limitations and Non-Claims

- Non-claim about tracking: "to avoid point tracking, we propose a novel Point 4D Transformer" (Abstract).
- Limitations or future work: Not specified.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single-modality point cloud videos ("point cloud videos") (Abstract; 1. Introduction).
> - Task structure: Two supervised tasks, "3D action recognition" (video-level classification) and "4D semantic segmentation" (point-level classification) (1. Introduction; 3.3).
> - Representation rigidity: Fixed-size clips and fixed point sampling in experiments ("fixed number of frames"; "we sample 2,048 points for each frame") (4.1).
> - Model sharing vs specialization: Task-specific prediction heads ("MLP layer converts the global feature to action predictions" vs "MLP layer that converts point features to point predictions"); weight sharing across tasks not specified (3.3).
> - Role of positional encoding: Absolute 4D coordinate embedding at input ("Therefore, we combine anchor coordinates, *i.e.*, (x, y, z, t), and local area features as the input to our transformer,") (3.2.1).

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates two tasks, "3D action recognition" and "4D semantic segmentation" (1. Introduction; 3.3), which makes it multi-task. All evaluations are on point cloud video datasets ("point cloud videos"; MSR-Action3D, NTU RGB+D 60/120, Synthia 4D), and no cross-domain transfer is claimed (Abstract; 1. Introduction; 4.1; 4.2).
