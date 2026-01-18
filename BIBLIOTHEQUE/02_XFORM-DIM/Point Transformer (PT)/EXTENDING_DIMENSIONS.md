## 1. Basic Metadata

- Title: "Point Transformer" (Title)
- Authors: "Hengshuang Zhao<sup>1,2</sup> Li Jiang<sup>3</sup> Jiaya Jia<sup>3</sup> Philip Torr<sup>1</sup> Vladlen Koltun<sup>4</sup>" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper's primary contribution is to "design self-attention layers for point clouds" and build Point Transformer networks for 3D point-cloud tasks such as segmentation and classification (Abstract).

## 3. Tasks Evaluated

### Task 1: Semantic scene segmentation
- Task type: Segmentation
- Dataset(s): S3DIS
- Domain: 3D point clouds of indoor scenes
- Evidence: "tasks such as semantic scene segmentation, object part segmentation, and object classification" (Abstract); "For 3D semantic segmentation, we use the challenging Stanford Large-Scale 3D Indoor Spaces (S3DIS) dataset [1]." (Section 4. Experiments); "The S3DIS [1] dataset for semantic scene parsing consists of 271 rooms in six areas from three different buildings." (Section 4.1. Semantic Segmentation); "3D point clouds are sets embedded in continuous space." (Introduction)

### Task 2: 3D shape classification (object classification)
- Task type: Classification
- Dataset(s): ModelNet40
- Domain: 3D CAD models / 3D point clouds
- Evidence: "tasks such as semantic scene segmentation, object part segmentation, and object classification" (Abstract); "For 3D shape classification, we use the widely adopted ModelNet40 dataset [47]." (Section 4. Experiments); "The ModelNet40 [47] dataset contains 12,311 CAD models with 40 object categories." (Section 4.2. Shape Classification)

### Task 3: Object part segmentation
- Task type: Segmentation
- Dataset(s): ShapeNetPart
- Domain: 3D object shapes / 3D point clouds
- Evidence: "tasks such as semantic scene segmentation, object part segmentation, and object classification" (Abstract); "And for object part segmentation, we use ShapeNetPart [52]." (Section 4. Experiments); "The ShapeNetPart dataset [52] is annotated for 3D object part segmentation." (Section 4.3. Object Part Segmentation)

## 4. Domain and Modality Scope

- Scope of evaluation: Multiple domains within the same modality (3D point clouds). Evidence: "We evaluate the effectiveness of the presented Point Transformer design on a number of domains and tasks. For 3D semantic segmentation, we use the challenging Stanford Large-Scale 3D Indoor Spaces (S3DIS) dataset [1]. For 3D shape classification, we use the widely adopted ModelNet40 dataset [47]. And for object part segmentation, we use ShapeNetPart [52]." (Section 4. Experiments)
- Domain generalization / cross-domain transfer claim: Not claimed.

## 5. Model Sharing Across Tasks

Training is described per dataset with different schedules: "For semantic segmentation on S3DIS, we train for 40K iterations... For 3D shape classification on ModelNet40 and 3D object part segmentation on ShapeNetPart, we train for 200 epochs." (Section 4. Experiments, Implementation details)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Semantic scene segmentation | Not specified. | Not specified. | Yes. | "For semantic segmentation, the final decoder stage produces a feature vector for each point in the input point set. We apply an MLP to map this feature to the final logits." (Section 3.5. Network Architecture, Output head) |
| 3D shape classification | Not specified. | Not specified. | Yes. | "For classification, we perform global average pooling over the pointwise features to get a global feature vector for the whole point set. This global feature is passed through an MLP to get the global classification logits." (Section 3.5. Network Architecture, Output head) |
| Object part segmentation | Not specified. | Not specified. | Not specified. | Not specified. |

## 6. Input and Representation Constraints

- Input dimensionality (fixed 3D): "3D point clouds are sets embedded in continuous space." (Introduction); "Here $\mathbf{p}_i$ and $\mathbf{p}_j$ are the 3D point coordinates for points i and j." (Section 3.3. Position Encoding)
- Input resolution / number of points: Variable N with fixed downsampling ratios: "the cardinality of the point set produced by each stage is [N, N/4, N/16, N/64, N/256], where N is the number of input points." (Section 3.5. Network Architecture)
- Local neighborhood size (context for attention): "Here the subset $\mathcal{X}(i) \subseteq \mathcal{X}$ is a set of points in a local neighborhood (specifically, k nearest neighbors) of $\mathbf{x}_i$." (Section 3.2. Point Transformer Layer); "We use k = 16 throughout" (Section 3.5. Network Architecture, Transition down)
- Sampling requirement for some tasks: "we follow the data preparation procedure of Qi et al. [27] and uniformly sample the points from each CAD model together with the normal vectors from the object meshes." (Section 4.2. Shape Classification)
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified beyond variable N.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length (per attention neighborhood): k neighbors; "k nearest neighbors" with "k = 16" used throughout (Section 3.2. Point Transformer Layer; Section 3.5. Network Architecture)
- Fixed or variable sequence length: Fixed k for each local neighborhood, with variable overall set size N (Section 3.2. Point Transformer Layer; Section 3.5. Network Architecture)
- Attention type: Local/windowed (kNN) self-attention; "we adopt the practice of recent self-attention networks for image analysis in applying self-attention locally, within a local neighborhood around each datapoint" (Section 3.2. Point Transformer Layer)
- Computational cost management: Local attention instead of global; "we apply selfattention locally, which enables scalability to large scenes with millions of points" (Related Work, Transformer and self-attention)

## 8. Positional Encoding (Critical Section)

- Mechanism: Trainable relative position encoding from coordinate differences; "We go beyond this by introducing trainable, parameterized position encoding. Our position encoding function $\delta$ is defined as follows: $\delta = \theta(\mathbf{p}_i - \mathbf{p}_j)$." (Section 3.3. Position Encoding)
- Where applied: Added to both attention and feature branches; "We use the subtraction relation and add a position encoding $\delta$ to both the attention vector $\gamma$ and the transformed features $\alpha$" (Section 3.2. Point Transformer Layer); "Notably, we found that position encoding is important for both the attention generation branch and the feature transformation branch. Thus Eq. 3 adds the trainable position encoding in both branches." (Section 3.3. Position Encoding)
- Fixed across experiments or modified: Positional encoding is ablated and compared; "We now study the choice of the position encoding $\delta$. The results are shown in Table 6." (Section 4.4. Ablation Study)

## 9. Positional Encoding as a Variable

- Treated as a research variable: Yes; "We now study the choice of the position encoding $\delta$. The results are shown in Table 6." (Section 4.4. Ablation Study)
- Multiple positional encodings compared: Yes; Table 6 compares "none", "absolute", "relative", "relative for attention", and "relative for feature" (Table 6. Ablation study: position encoding.)
- Claims about PE criticality: "We can see that without position encoding, the performance drops significantly. With absolute position encoding, the performance is higher than without. Relative position encoding yields the highest performance." (Section 4.4. Ablation Study)

## 10. Evidence of Constraint Masking

- Model sizes: "The number of parameters in Point Transformer (4.9M) is much smaller than in current high-performing architectures such as KPConv (14.9M) and SparseConv (30.1M)." (Section 4.1. Semantic Segmentation)
- Dataset sizes: "The S3DIS [1] dataset for semantic scene parsing consists of 271 rooms in six areas from three different buildings." (Section 4.1. Semantic Segmentation); "The ModelNet40 [47] dataset contains 12,311 CAD models with 40 object categories." (Section 4.2. Shape Classification); "The ShapeNetPart dataset [52] is annotated for 3D object part segmentation. It consists of 16,880 models from 16 shape categories, with 14,006 3D models for training and 2,874 for testing." (Section 4.3. Object Part Segmentation)
- Attribution of gains: The paper emphasizes architectural choices rather than scaling; "without position encoding, the performance drops significantly" and "Relative position encoding yields the highest performance" (Section 4.4. Ablation Study); "Vector attention is more expressive... This expressivity appears to be very beneficial in 3D data processing." (Section 4.4. Ablation Study)

## 11. Architectural Workarounds

- Local attention to manage scale: "we adopt the practice... applying self-attention locally, within a local neighborhood around each datapoint" (Section 3.2. Point Transformer Layer); "we apply selfattention locally, which enables scalability to large scenes with millions of points" (Related Work, Transformer and self-attention)
- Hierarchical stages and downsampling: "The feature encoder... has five stages that operate on progressively downsampled point sets. The downsampling rates for the stages are [1, 4, 4, 4, 4]" (Section 3.5. Network Architecture)
- Transition down (sampling + pooling): "We perform farthest point sampling [27]..." and "we use a kNN graph... followed by max pooling" (Section 3.5. Network Architecture, Transition down)
- Transition up (interpolation + skip connections): "we adopt a U-net design..." and "mapped onto the higher-resolution point set... via trilinear interpolation... provided via a skip connection" (Section 3.5. Network Architecture, Transition up)
- Dimensionality reduction in blocks: "linear projections that can reduce dimensionality and accelerate processing" (Section 3.4. Point Transformer Block)

## 12. Explicit Limitations and Non-Claims

- Training caveat: "Note that we did not use loss-balancing during training, which can boost category mIoU." (Section 4.3. Object Part Segmentation)
- Future work direction: "We hope that our work will inspire further investigation of the properties of point transformers, the development of new operators and network designs, and the application of transformers to other tasks, such as 3D object detection." (Section 5. Conclusion)
- Open-world learning / unrestrained multi-task learning / meta-learning: Not specified.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Multiple 3D point-cloud domains/datasets within a single modality (Section 4. Experiments)
- Task structure: Three supervised tasks (semantic scene segmentation, 3D shape classification, object part segmentation) (Abstract; Section 4. Experiments)
- Representation rigidity: 3D point coordinates with kNN local neighborhoods (k=16) and fixed downsampling ratios across stages (Section 3.2; Section 3.5)
- Model sharing vs specialization: Weight sharing across tasks not specified; task-specific heads are described for segmentation and classification (Section 3.5)
- Role of positional encoding: Trainable relative position encoding is central and ablated; relative PE performs best (Section 3.3; Section 4.4)

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates multiple tasks across different 3D point-cloud domains: "semantic scene segmentation" on S3DIS, "3D shape classification" on ModelNet40, and "object part segmentation" on ShapeNetPart (Abstract; Section 4. Experiments). These are distinct datasets/domains within the same modality, and no cross-domain transfer or open-world claims are made, so the setup remains constrained to specific supervised benchmarks.
