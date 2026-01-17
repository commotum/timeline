## 1. Basic Metadata

- Title: "OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving" (Title)
- Authors: "Wenzhao Zheng<sup>1,2\*</sup>, Weiliang Chen<sup>1\*</sup>, Yuanhui Huang<sup>1</sup>, Borui Zhang<sup>1</sup>, Yueqi Duan<sup>1</sup>, and Jiwen Lu<sup>1†</sup>" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper introduces "a new framework of learning a world model, OccWorld, in the 3D occupancy space to simultaneously predict the movement of the ego car and the evolution of the surrounding scenes" (Abstract).

## 3. Tasks Evaluated

### Task 1

- Task name: 4D occupancy forecasting
- Task type: Segmentation; Reconstruction
- Dataset(s) used: Occ3D [53] (nuScenes)
- Domain: 3D semantic occupancy grids of autonomous driving scenes
- Quotes: "We conduct two tasks to evaluate our OccWorld: 4D occupancy forecasting on Occ3D [53] and motion planning on nuScenes [3]." (Section 4.1 Task Descriptions); "In this paper, we explore 4D occupancy forecasting, which aims to forecast future 3D occupancy given historical occupancy." (Section 4.1 Task Descriptions); "Occ3D [53] provides 3D semantic occupancy annotations for nuScenes. Each scene is split into  $200 \times 200 \times 16$  voxels covering a -40m  $\sim$  40m area along the X and Y axis and -1m  $\sim$  5.4m along the Z axis." (Section 4.2 Datasets Details)

### Task 2

- Task name: Motion planning
- Task type: Other (trajectory planning)
- Dataset(s) used: nuScenes [3]
- Domain: BEV trajectory planning in autonomous driving scenes
- Quotes: "We conduct two tasks to evaluate our OccWorld: 4D occupancy forecasting on Occ3D [53] and motion planning on nuScenes [3]." (Section 4.1 Task Descriptions); "Motion planning aims to produce safe future trajectories for the vehicle given ground-truth surrounding information or perception results. The planned trajectory is represented by a series of 2D waypoints in the BEV plane." (Section 4.1 Task Descriptions); "nuScenes [3] contains 1000 driving scenes" (Section 4.2 Datasets Details)

## 4. Domain and Modality Scope

- Single domain: Yes. Evidence: "Occ3D [53] provides 3D semantic occupancy annotations for nuScenes." (Section 4.2 Datasets Details); "nuScenes [3] contains 1000 driving scenes" (Section 4.2 Datasets Details).
- Multiple domains within the same modality: Not indicated; evaluations are on nuScenes/Occ3D driving scenes (Section 4.2 Datasets Details).
- Multiple modalities: Yes. Evidence: "nuScenes [3] contains 1000 driving scenes, i.e., videos of 6 surrounding cameras with 360° horizontal FOV and 32-beam LiDAR point clouds" (Section 4.2 Datasets Details); "OccWorld-D, OccWorld-T, and OccWorld-S can be seen as end-to-end vision-based 4D occupancy forecasting methods as they take surrounding images as input." (Section 4.4 Results and Analysis).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 4D occupancy forecasting | Yes (shared world model for scene/ego prediction) | Not specified | Yes (scene decoder) | "a new framework of learning a world model, OccWorld, in the 3D occupancy space to simultaneously predict the movement of the ego car and the evolution of the surrounding scenes." (Abstract); "We then adopt a GPT-like spatial-temporal generative transformer to generate subsequent scene and ego tokens to decode the future occupancy and ego trajectory." (Abstract); "we reuse the scene decoder d to decode the predicted 3D occupancy ... and additionally learn an ego decoder d_{ego} to produce the ego displacement" (Section 3.4 OccWorld: a 3D Occupancy World Model). |
| Motion planning | Yes (shared world model for scene/ego prediction) | Not specified | Yes (ego decoder) | "a new framework of learning a world model, OccWorld, in the 3D occupancy space to simultaneously predict the movement of the ego car and the evolution of the surrounding scenes." (Abstract); "We then adopt a GPT-like spatial-temporal generative transformer to generate subsequent scene and ego tokens to decode the future occupancy and ego trajectory." (Abstract); "we reuse the scene decoder d to decode the predicted 3D occupancy ... and additionally learn an ego decoder d_{ego} to produce the ego displacement" (Section 3.4 OccWorld: a 3D Occupancy World Model). |

## 6. Input and Representation Constraints

- 3D occupancy is a fixed voxel grid representation: "we propose to adopt 3D occupancy as the 3D scene representation  $\mathbf{y} \in \mathbb{R}^{H \times W \times D}$ . 3D occupancy partitions the surrounding 3D space into  $H \times W \times D$  voxels" (Section 3.2 3D Occupancy Scene Tokenizer).
- Occ3D grid size and spatial extents: "Each scene is split into  $200 \times 200 \times 16$  voxels covering a -40m  $\sim$  40m area along the X and Y axis and -1m  $\sim$  5.4m along the Z axis." (Section 4.2 Datasets Details).
- BEV conversion for efficiency: "we first transform the 3D occupancy  $\mathbf{y} \in \mathbb{R}^{H \times W \times D}$  to a BEV representation  $\hat{\mathbf{y}} \in \mathbb{R}^{H \times W \times DC'}$" (Section 3.2 3D Occupancy Scene Tokenizer).
- Fixed down-sampling factor in tokenizer: "obtain downsampled features  $\hat{\mathbf{z}} \in \mathbb{R}^{\frac{H}{d} \times \frac{W}{d} \times C}$ , where d is the down-sampling factor." (Section 3.2 3D Occupancy Scene Tokenizer); "The scene tokenizer employs a down-sampling factor of 4, featuring a codebook with a size of 512 and a dimension of 128." (Section 4.3 Implementation Details).
- Planning output representation: "The planned trajectory is represented by a series of 2D waypoints in the BEV plane." (Section 4.1 Task Descriptions).

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; the experiments use "a 2-second historical context to forecast the subsequent 3 seconds." (Section 4.3 Implementation Details).
- Fixed or variable sequence length: Not specified.
- Attention type: "spatial-wise temporal causal self-attention" (Fig. 3 caption); "TA denotes masked temporal attention which blocks the effect of future tokens to previous tokens." (Section 3.3 Spatial-Temporal Generative Transformer); "We apply spatial aggregation (e.g., self-attention [8])" (Section 3.3 Spatial-Temporal Generative Transformer).
- Mechanisms to manage computational cost: "We then merge the scene tokens in each  $2 \times 2$  window with a stride of 2 to achieve a 1/4 down-sampling. We repeat this procedure for K times to obtain world tokens of hierarchical scales" (Section 3.3 Spatial-Temporal Generative Transformer); "We finally employ a U-net structure to aggregate predicted tokens at different scales to ensure spatial consistency." (Section 3.3 Spatial-Temporal Generative Transformer).

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Not specified.
- Where it is applied: Not specified.
- Fixed across experiments / modified per task / ablated: Not specified.

## 9. Positional Encoding as a Variable

- Treated as a core research variable or fixed assumption: Not specified.
- Multiple positional encodings compared: Not specified.
- Claims PE choice is not critical or secondary: Not specified.

## 10. Evidence of Constraint Masking

- Dataset size: "nuScenes [3] contains 1000 driving scenes" and "we employ 700 and 150 scenes for training and validation, respectively." (Section 4.2 Datasets Details).
- Model size / capacity indicators: "The scene tokenizer employs a down-sampling factor of 4, featuring a codebook with a size of 512 and a dimension of 128." (Section 4.3 Implementation Details); "The spatial-temporal generative transformer comprises 3 scales, each incorporating 6 layers of spatial-wise temporal attention" (Section 4.3 Implementation Details).
- Performance gains attributed to architecture: "We observe that using spatial aggregation to model spatial dependencies and using temporal attention to integrate history information is vital to the performance of both 4D occupancy forecasting and motion planning tasks." (Section 4.4 Results and Analysis).
- Hyperparameter sensitivity: "We see that using a larger codebook than 512 leads to overfitting and using a smaller S, C, and N might not be enough to capture the scene distribution." (Section 4.4 Results and Analysis).
- Scaling model size or data as the primary driver of gains: Not explicitly claimed.

## 11. Architectural Workarounds

- Scene tokenizer with discrete tokens for compact representation: "We train a vector-quantized autoencoder (VQ-VAE) [42] on  $\bf y$  to obtain discrete tokens  $\bf z$  to better represent the scene" (Section 3.2 3D Occupancy Scene Tokenizer).
- BEV conversion and 2D conv encoder for efficiency: "For efficiency, we first transform the 3D occupancy ... to a BEV representation ... We then adopt a lightweight encoder composed of 2D convolutions" (Section 3.2 3D Occupancy Scene Tokenizer).
- Hierarchical multi-scale tokens via window merging: "We then merge the scene tokens in each  $2 \times 2$  window with a stride of 2 to achieve a 1/4 down-sampling. We repeat this procedure for K times to obtain world tokens of hierarchical scales" (Section 3.3 Spatial-Temporal Generative Transformer).
- Temporal causal attention: "TA denotes masked temporal attention which blocks the effect of future tokens to previous tokens." (Section 3.3 Spatial-Temporal Generative Transformer).
- U-net aggregation for multi-scale consistency: "We finally employ a U-net structure to aggregate predicted tokens at different scales to ensure spatial consistency." (Section 3.3 Spatial-Temporal Generative Transformer).
- Separate decoders for scene vs ego outputs: "we reuse the scene decoder d to decode the predicted 3D occupancy ... and additionally learn an ego decoder d_{ego} to produce the ego displacement" (Section 3.4 OccWorld: a 3D Occupancy World Model).

## 12. Explicit Limitations and Non-Claims

- Limitation / future work: "OccWorld simultaneously models the ego movements and scene evolutions, yet cannot predict the futures conditioned on certain driving commands. However, the ability to forecast multiple futures based on different conditions is important for a world model and is an interesting future direction." (Limitations).
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single autonomous driving domain (nuScenes/Occ3D), no cross-domain claims.
> – Task structure: Two tasks (4D occupancy forecasting and motion planning) evaluated within the same domain.
> – Representation rigidity: Fixed voxel-grid occupancy with BEV conversion and fixed tokenizer hyperparameters.
> – Model sharing vs specialization: Shared world model with separate scene and ego decoders.
> – Role of positional encoding: Not specified in the OCR text.

### 14. Final Classification

Classification: **Multi-task, single-domain**.

The paper "conduct two tasks to evaluate our OccWorld: 4D occupancy forecasting on Occ3D [53] and motion planning on nuScenes [3]" (Section 4.1 Task Descriptions), and both tasks are evaluated on the nuScenes driving domain (Section 4.2 Datasets Details). It does not claim cross-domain transfer or evaluation beyond nuScenes/Occ3D.
