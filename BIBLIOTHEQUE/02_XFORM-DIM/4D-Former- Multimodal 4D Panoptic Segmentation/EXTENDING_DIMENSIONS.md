## 1. Basic Metadata

- Title: "4D-Former: Multimodal 4D Panoptic Segmentation" (Title)
- Authors: "Ali Athar"; "Enxu Li"; "Sergio Casas"; "Raquel Urtasun" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes 4D-Former, "a novel method for 4D panoptic segmentation which leverages both LiDAR and image modalities" to output semantic and temporally consistent object masks for LiDAR point-cloud sequences (Abstract).

## 3. Tasks Evaluated

- Task name: 4D panoptic segmentation (panoptic tracking)
  - Task type: Segmentation; Tracking
  - Dataset(s) used: nuScenes; SemanticKITTI
  - Domain: LiDAR point-cloud sequences with RGB camera images (autonomous driving)
  - Quotes: "4D panoptic segmentation is a challenging but practically useful task that requires every point in a LiDAR point-cloud sequence to be assigned a semantic class label, and individual objects to be segmented and tracked over time." (Abstract) "We apply 4D-Former to the nuScenes and SemanticKITTI datasets where it achieves state-of-the-art results." (Abstract) "The scenes are captured with a 32-beam LiDAR sensor and 6 cameras mounted at different angles around the ego vehicle." (4 Experiments, Datasets)

- Task name: 3D panoptic segmentation (reported metrics)
  - Task type: Segmentation
  - Dataset(s) used: nuScenes; SemanticKITTI
  - Domain: LiDAR point clouds (autonomous driving)
  - Quotes: "we first present the 3D panoptic metrics on the two benchmarks for reference" (Supplementary Materials, B Detailed Quantitative Results) "we evaluate our SemanticKITTI results using 3D panoptic metrics" (Supplementary Materials, B Detailed Quantitative Results)

## 4. Domain and Modality Scope

- Evaluation is performed on a single domain (autonomous driving scenes) across two datasets: "The scenes are captured with a 32-beam LiDAR sensor and 6 cameras mounted at different angles around the ego vehicle." (4 Experiments, Datasets) "SemanticKITTI [9] contains fewer but longer sequences" (4 Experiments, Datasets)
- Multiple modalities are used: "we propose 4D-Former: a novel method for 4D panoptic segmentation which leverages both LiDAR and image modalities" (Abstract)
- Multiple domains within the same modality: Not indicated beyond two autonomous driving datasets (no explicit cross-domain claim).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 4D panoptic segmentation | Yes (single unified decoder for semantic + track masks) | Not specified. | No separate heads for semantic/track masks. | "We propose a novel decoder which predicts per-point semantic and object track masks with a unified architecture. This stands in contrast with existing methods [9, 11, 48, 6] which generally have separate heads for each output." (3.2 Transformer-based Panoptic Decoder) |
| 3D panoptic segmentation (reported metrics) | Not specified (same model outputs evaluated with 3D metrics). | Not specified. | Not specified. | "we first present the 3D panoptic metrics on the two benchmarks for reference" (Supplementary Materials, B Detailed Quantitative Results) "we evaluate our SemanticKITTI results using 3D panoptic metrics" (Supplementary Materials, B Detailed Quantitative Results) |

## 6. Input and Representation Constraints

- Fixed clip length: "We process clips containing 2 frames each" (4 Experiments, Implementation Details)
- Sliding-window inputs: "4D-Former operates in a sliding window fashion" and "At each iteration, 4D-Former takes as input the current LiDAR scan at time t, the past scan at t-1, and the camera images at time t." (3 Multimodal 4D Panoptic Segmentation)
- Image resolution constraint: "Assume the driving scene is captured by a set of images of size  $H \times W$" (3.1 Multimodal Encoder) and "The images are resized in an aspect-ratio preserving manner such that the lower dimension is 480px." (4 Experiments, Implementation Details)
- Point sampling constraint: "we randomly subsample the LiDAR pointcloud to  $10^5$  points" (4 Experiments, Implementation Details)
- Point representation: "Each of the N points in the input LiDAR point-cloud is represented as an 8-D feature which include the xyz coordinates, relative timestamp, intensity, and 3D relative offsets to the nearest voxel center." (3.1 Multimodal Encoder)
- Voxel/grid resolution: "with voxel size of 0.1 m" (4 Experiments, Implementation Details)
- Feature dimensionality: "the feature dimensionality D=128" (4 Experiments, Implementation Details)
- Fixed number of queries / tokens: "the number of queries (T) is assumed to be an upper-bound on the number of objects in a given scene." (3.2 Transformer-based Panoptic Decoder)
- Fixed patch size: Not specified.
- Padding requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; processing uses 2-frame clips: "We process clips containing 2 frames each" (4 Experiments, Implementation Details)
- Fixed or variable length: Clip length is fixed to 2 frames, but overall streams can be arbitrary length via sliding window: "In order to handle sequences of arbitrary length as well as continuous streams of data (e.g., in the onboard setting), 4D-Former operates in a sliding window fashion" (3 Multimodal 4D Panoptic Segmentation)
- Attention type: Global cross-attention + self-attention over queries: "cross-attending to the voxel features" and "cross-attending to the set of image features" and "self-attending to each other twice intermittently" (3.2 Transformer-based Panoptic Decoder). "cross-attention allows them to learn global context by attending to the features from both modalities across the entire scene." (3.2 Transformer-based Panoptic Decoder)
- Computational cost management: "This mitigates the need for dense feature interaction between the two modalities which, if done naively, would be computationally intractable since  $N_i$  and  $M_i$  are on the order of  $10^4$" and "Our fusion block avoids this by leveraging a set of concise queries which attend to the scene features from both modalities" with "$T \ll N_i$" and "$T \ll M_i$" (3.2 Transformer-based Panoptic Decoder)

## 8. Positional Encoding (Critical Section)

- Mechanism: "We impart the cross-attention operation with 3D coordinate information of the features in  V_i  by using positional encodings (E in Eq. 3). These contain two components: (1) Fourier encodings [51] of the (x,y,z) coordinates, and (2) a depth component which is obtained by applying sine and cosine activations at various frequencies to the Euclidean distance of each voxel feature from the LiDAR sensor." (3.2 Transformer-based Panoptic Decoder, Positional Encodings)
- Where applied: The encodings are added inside cross-attention: "Q(K+E)^{\text{T}}" (Eq. 3) and "We impart the cross-attention operation with 3D coordinate information" (3.2 Transformer-based Panoptic Decoder, Positional Encodings)
- Image feature positional encoding: "For the image features  $\mathcal{F}_i$ , we use the encoding of the corresponding voxel." (3.2 Transformer-based Panoptic Decoder, Positional Encodings)
- Fixed vs modified: Depth component is ablated in experiments: "Row 4 omits this component and instead only uses Fourier encodings based on the xyz coordinates." (4 Experiments, Effect of Depth Encodings)

## 9. Positional Encoding as a Variable

- The paper treats positional encoding as a variable in ablations (depth component removed vs. full): "our positional encodings contain a depth component" and "Row 4 omits this component and instead only uses Fourier encodings based on the xyz coordinates." (4 Experiments, Effect of Depth Encodings)
- Multiple positional encodings compared: Yes (full Fourier+depth vs. Fourier-only). (4 Experiments, Effect of Depth Encodings)
- Claims that PE choice is "not critical" or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): "nuScenes [12, 53] contains 1000 sequences, each 20s long" and "The training set contains 600 sequences, whereas validation and test each contain 150." (4 Experiments, Datasets) "SemanticKITTI [9] contains fewer but longer sequences" (4 Experiments, Datasets)
- Performance gains attributed to architecture/training rather than scaling: "We attribute this to 4D-Former's ability to reason over multimodal inputs and segment both semantic classes and object tracks in an end-to-end learned fashion." (4 Experiments, Comparison to state-of-the-art) "This shows that using image information yields significant performance improvements." (4 Experiments, Effect of Image Fusion)
- Training tricks: "Inspired by [50, 28], we employ soft-masked cross-attention to improve convergence." (3.2 Transformer-based Panoptic Decoder)
- Data scaling notes: "For these experiments, we subsample the training set by using only every fourth frame to save time and resources." (4 Experiments, Ablations)

## 11. Architectural Workarounds

- Sliding-window processing for long sequences: "4D-Former operates in a sliding window fashion" (3 Multimodal 4D Panoptic Segmentation)
- Query-based fusion to reduce interaction cost: "We initialize a set of queries" and "Our fusion block avoids this by leveraging a set of concise queries which attend to the scene features from both modalities" (3.2 Transformer-based Panoptic Decoder)
- Point/voxel dual-branch with sparse convolutions: "consists of a point-branch and a voxel-branch" and "3D sparse convolutional blocks" (3.1 Multimodal Encoder)
- Soft-masked cross-attention for convergence: "we employ soft-masked cross-attention to improve convergence" (3.2 Transformer-based Panoptic Decoder)
- Tracklet Association Module + memory bank for long-term tracking: "we propose a learnable Tracklet Association Module (TAM) which can associate tracklets across longer frame gaps" and "we maintain a memory bank containing object tracks" (3.3 Tracklet Association Module)

## 12. Explicit Limitations and Non-Claims

- Limitations: "Our method performs less effectively on SemanticKITTI compared to nuScenes, particularly in crowded scenes with several objects." (5 Limitations) "our tracking quality is generally good for vehicles, but is comparatively worse for smaller object classes e.g. bicycle, pedestrian" and "the improvement plateaus at  $T_{\rm hist}=4$" (5 Limitations)
- Future work: "a promising future direction is to develop more effective augmentation techniques for multimodal training." (5 Limitations) "Another area for future work thus involves improving the tracking mechanism to handle longer time horizons and challenging object classes." (5 Limitations)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

## 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single domain (autonomous driving), evaluated on nuScenes and SemanticKITTI with LiDAR + RGB cameras.
> – Task structure: 4D panoptic segmentation (semantic + instance tracking) with additional 3D panoptic metrics for reference.
> – Representation rigidity: Fixed 2-frame clips, fixed voxel size (0.1 m), fixed feature dimension (D=128), fixed query count upper-bound, image resizing and point subsampling.
> – Model sharing vs specialization: Unified decoder for semantic and track masks; TAM adds a specialized tracking association module.
> – Role of positional encoding: Explicit Fourier+depth encodings in cross-attention, with depth ablated in experiments.

## 14. Final Classification

**Multi-task, single-domain.** The task explicitly combines semantic labeling and tracking within "4D panoptic segmentation" (Abstract) and is evaluated on autonomous driving LiDAR+camera datasets (4 Experiments, Datasets), which keeps the domain fixed. The paper does not claim cross-domain transfer or unrestrained multi-domain evaluation.
