## 1. Basic Metadata
- Title: "Uni3DL: Unified Model for 3D and Language Understanding" (Paper header)
- Authors: "Xiang Li<sup>1,\*</sup>, Jian Ding<sup>1,\*</sup>, Zhaoyang Chen<sup>2</sup>, Mohamed Elhoseiny<sup>1</sup>" (Paper header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
The paper presents "Uni3DL, a unified model for 3D and Language understanding" that "operates directly on point clouds" to provide a unified architecture across diverse 3D vision-language tasks (Abstract).

## 3. Tasks Evaluated
Task name: 3D semantic segmentation
Task type: Segmentation
Dataset(s) used: ScanNet (v2); S3DIS
Domain: 3D indoor RGB-D scans / point clouds
Quotes: "Uni3DL has been rigorously evaluated across diverse 3D vision-language understanding tasks, including semantic segmentation" (Abstract); "For model evaluation, other than ScanNet (v2), ScanRefer, Cap3D, we use additional S3DIS [2] to evaluate both semantic and instance segmentation" (Section 4.1 Dataset); "ScanNet (v2) [20] captures RGB-D videos with 2.5 million views from more than 1,500 3D scans." (Section 4.1 Dataset)

Task name: 3D instance segmentation
Task type: Segmentation
Dataset(s) used: ScanNet (v2); S3DIS
Domain: 3D indoor RGB-D scans / point clouds
Quotes: "Uni3DL has been rigorously evaluated across diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, visual grounding, 3D captioning, and text-3D cross-modal retrieval." (Abstract); "We pretrain our Uni3DL on three datasets, including Scan-Net (v2) [20] for instance segmentation, ScanRefer [9] for visual grounding, and Cap3D Objaverse [45] dataset for 3D captioning and text-3D cross-modal retrieval." (Section 4.1 Dataset); "For model evaluation, other than ScanNet (v2), ScanRefer, Cap3D, we use additional S3DIS [2] to evaluate both semantic and instance segmentation, Text2Shape [12] to evaluate text-to-3D retrieval." (Section 4.1 Dataset)

Task name: 3D object detection
Task type: Detection
Dataset(s) used: ScanNet (v2) (SN Val)
Domain: 3D indoor RGB-D scans / point clouds
Quotes: "We compare 3D semantic segmentation, object detection, and instance segmentation performance with previous STOA methods in Table 3." (Section 4.3 3D Semantic/Instance Sementation); "A task router module with multiple functional heads is faithfully designed to support diverse vision-language tasks, including 3D object detection" (Conclusion); "Table 3. Performance of our Uni3DL on different segmentation and VL tasks. 'SN' denotes the ScanNet (v2) dataset." (Table 3 caption)

Task name: 3D grounded segmentation / visual grounding
Task type: Segmentation; Reasoning / relational
Dataset(s) used: ScanRefer
Domain: 3D indoor scenes with referring language
Quotes: "Uni3DL has been rigorously evaluated across diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, visual grounding, 3D captioning, and text-3D cross-modal retrieval." (Abstract); "We compare the 3D grounded segmentation performance of our Uni3DL" (Section 4.4 3D Visual Grounding); "ScanRefer [9] dataset contains 51,583 referring descriptions of 11,046 objects from 800 ScanNet scenes." (Section 4.1 Dataset)

Task name: 3D captioning
Task type: Generation
Dataset(s) used: Cap3D Objaverse
Domain: 3D object shapes with text descriptions
Quotes: "Uni3DL has been rigorously evaluated across diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, visual grounding, 3D captioning, and text-3D cross-modal retrieval." (Abstract); "We pretrain our Uni3DL on three datasets, including Scan-Net (v2) [20] for instance segmentation, ScanRefer [9] for visual grounding, and Cap3D Objaverse [45] dataset for 3D captioning and text-3D cross-modal retrieval." (Section 4.1 Dataset); "From Table 3, our Uni3DL model outperforms existing methods in 3D captioning on the Cap3D Objaverse dataset" (Section 4.5 3D Captioning)

Task name: Text-to-3D retrieval / text-3D cross-modal retrieval
Task type: Other (retrieval)
Dataset(s) used: Cap3D Objaverse; Text2Shape
Domain: 3D object shapes with text
Quotes: "Uni3DL has been rigorously evaluated across diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, visual grounding, 3D captioning, and text-3D cross-modal retrieval." (Abstract); "We pretrain our Uni3DL on three datasets, including Scan-Net (v2) [20] for instance segmentation, ScanRefer [9] for visual grounding, and Cap3D Objaverse [45] dataset for 3D captioning and text-3D cross-modal retrieval." (Section 4.1 Dataset); "We evaluate text-to-3D retrieval performance on the Text2Shape ShapeNet subset." (Section 4.6 Text-to-3D Retrieval)

Task name: 3D object classification / shape classification (zero-shot)
Task type: Classification
Dataset(s) used: ModelNet40; ModelNet10 (evaluation); Cap3D Objaverse (fine-tuning source)
Domain: 3D CAD models
Quotes: "The Uni3DL is a versatile architecture tailored for diverse 3D vision-language tasks, including 3D object classification" (Section 3.1 Method overview); "We use our Uni3DL model fine-tuned on the Cap3D Objaverse dataset to evaluate zero-shot 3D classification performance on ModelNet40 and ModelNet10 datasets." (Section B.1 Zero-Shot 3D Classification); "ModelNet40 includes 40 different categories with 12, 311 CAD models, while ModelNet10, a smaller subset, consists of 10 categories with 4, 899 models." (Section B.1 Zero-Shot 3D Classification)

Task name: Grounded localization
Task type: Detection
Dataset(s) used: ScanRefer
Domain: 3D indoor scenes with referring language
Quotes: "Previous methods have also explored the grounded localization task." (Section B.2 Grounded Localization); "To produce grounded object location, we directly use grounded object masks to calculate their bounding boxes." (Section B.2 Grounded Localization); "Table 7. Comparative analysis of grounded localization performance on the ScanRefer [9] dataset." (Table 7 caption)

## 4. Domain and Modality Scope
- Is evaluation performed on a single domain? No; evaluation spans multiple 3D datasets including indoor scans and object datasets: "We pretrain our Uni3DL on three datasets, including Scan-Net (v2) [20] for instance segmentation, ScanRefer [9] for visual grounding, and Cap3D Objaverse [45] dataset for 3D captioning and text-3D cross-modal retrieval." and "For model evaluation, other than ScanNet (v2), ScanRefer, Cap3D, we use additional S3DIS [2] to evaluate both semantic and instance segmentation, Text2Shape [12] to evaluate text-to-3D retrieval." (Section 4.1 Dataset).
- Multiple domains within the same modality? Yes; indoor RGB-D scans (ScanNet/S3DIS/ScanRefer) and object-level shapes (Cap3D Objaverse/Text2Shape/ModelNet): "ScanNet (v2) [20] captures RGB-D videos" and "Cap3D Objaverse [45] dataset, is derived from Objaverse" and "Text2Shape [12] contains 8,447 table instances and 6,591 chair instances from the ShapeNet dataset" (Section 4.1 Dataset).
- Multiple modalities? Yes; point clouds and language: "a unified model for 3D and Language understanding" (Abstract) and the architecture includes "a **Text Encoder** for textual feature extraction; a **Point Encoder** dedicated to point feature learning" (Section 3.1 Method overview).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed; the paper reports a zero-shot evaluation but does not frame it as domain generalization: "We use our Uni3DL model fine-tuned on the Cap3D Objaverse dataset to evaluate zero-shot 3D classification performance on ModelNet40 and ModelNet10 datasets." (Section B.1 Zero-Shot 3D Classification)

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 3D semantic segmentation | Yes | Yes | Yes | "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract); "Finetuning for 3D semantic/instance Segmentation." (Section A.2 Finetuning); "**Mask Head.** Given mask output  $\\mathbf{O}^m \\in \\mathbb{R}^{Q \\times C}$ , and full-resolution voxel features  $\\mathbf{V}_s \\in \\mathbb{R}^{N_0 \\times C}$ , we calculate voxel mask as  $\\mathbf{O}_m = \\mathbf{O}^m \\cdot \\mathbf{V}_s^T$ ." (Section 3.4 Task Router) |
| 3D instance segmentation | Yes | Yes | Yes | "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract); "Finetuning for 3D semantic/instance Segmentation." (Section A.2 Finetuning); "the 3D instance segmentation task includes two heads, object classification, and mask prediction." (Section 3.4 Task Router) |
| 3D object detection | Yes | Not specified | Not specified | "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract); "including 3D object detection" (Conclusion) |
| 3D grounded segmentation / visual grounding | Yes | Yes | Yes | "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract); "Finetuning for Grounded Segmentation" (Section A.2 Finetuning); "**Grounding Head.** Visual grounding requires matching text descriptions to visual objects." (Section 3.4 Task Router) |
| 3D captioning | Yes | Yes | Yes | "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract); "Finetuning for 3D Captioning." (Section A.2 Finetuning); "**Text Generation Head.** In the context of 3D captioning, our method begins by generating textural embeddings for each token within the vocabulary" (Section 3.4 Task Router) |
| Text-to-3D retrieval / text-3D cross-modal retrieval | Yes | Yes | Yes | "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract); "Finetuning for Text-3D Cross-Modal Retrieval." (Section A.2 Finetuning); "**Text-3D Matching Head.** Our Uni3DL uses decoupled point and text encoder networks." (Section 3.4 Task Router) |
| 3D object classification / shape classification | Yes | Yes (fine-tuned on Cap3D before zero-shot eval) | Yes | "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract); "We use our Uni3DL model fine-tuned on the Cap3D Objaverse dataset to evaluate zero-shot 3D classification performance" (Section B.1 Zero-Shot 3D Classification); "**Object Classification Head.** We select the first Q output semantic outputs for object classification." (Section 3.4 Task Router) |
| Grounded localization | Yes (derived from grounded masks) | Not specified | Not specified | "To produce grounded object location, we directly use grounded object masks to calculate their bounding boxes." (Section B.2 Grounded Localization); "enjoys seamless task decomposition and substantial parameter sharing across tasks." (Abstract) |

## 6. Input and Representation Constraints
- Point cloud voxelization and 3D input assumption: "A colored input point cloud, denoted as  $\mathbf{P} \in \mathbb{R}^{N_0 \times 6}$ , undergoes quantization into  $N_0$  voxels represented as  $\mathbf{V}_0 \in \mathbb{R}^{N_0 \times 3}$" (Section 3.2 Point Cloud and Text Encoder).
- Fixed voxel size for scans/shapes: "During pretraining, the voxel size is set to 0.02m for 3D scans (e.g., ScanNet (v2)) and 0.01 for normalized 3D shapes (e.g., Cap3D Objaverse)" (Section 4.2 Implementation Details).
- Fixed-length input constraint via sampling: "Current transformer implementations generally require a fixed length of inputs in each batch entry. To enable efficient batch-wise training" (Section 3.3 Query Transformer Module).
- Fixed number of latent queries: "we employ 150 latent queries and an additional latent query for scene-level tasks." (Section 4.2 Implementation Details).
- Cropping/minimum points for S3DIS finetuning: "we randomly crop  $5m \times 5m \times 5m$  blocks from each scene, ensuring a minimum of 25,000 points per scene." (Section A.2 Finetuning).
- Fixed patch size / padding / resizing for text tokens: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified; the paper states "we employ 150 latent queries and an additional latent query for scene-level tasks" (Section 4.2 Implementation Details) and notes a fixed-length requirement for voxel inputs: "Current transformer implementations generally require a fixed length of inputs in each batch entry" (Section 3.3 Query Transformer Module).
- Fixed or variable length: Variable point clouds are sampled to a fixed length for batching: "Point clouds in a batch usually have different numbers of points, leading to differing voxel quantities. Current transformer implementations generally require a fixed length of inputs in each batch entry." (Section 3.3 Query Transformer Module).
- Attention type: Cross-attention and self-attention with masked attention: "a sequence of cross-attention and self-attention operations between latent queries, text queries and voxel features" (Figure 2 caption); "use masked attention instead of vanilla cross-attention where each query only attends to masked voxels predicted by the previous layer." (Section 3.3 Query Transformer Module).
- Mechanisms to manage computational cost: Voxel sampling and masked attention are used: "To enable efficient batch-wise training" (Section 3.3 Query Transformer Module) and "use masked attention instead of vanilla cross-attention" (Section 3.3 Query Transformer Module).

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism used: Not specified.
- Where it is applied: Not specified.
- Whether positional encoding is fixed/modified/ablated: Not specified.

## 9. Positional Encoding as a Variable
- Treated as a core research variable or fixed assumption: Not specified.
- Multiple positional encodings compared: Not specified.
- Claims that positional encoding choice is "not critical" or secondary: Not specified.

## 10. Evidence of Constraint Masking
- Model size(s): Model size not specified.
- Dataset size(s): "ScanNet (v2) [20] captures RGB-D videos with 2.5 million views from more than 1,500 3D scans." (Section 4.1 Dataset); "ScanRefer [9] dataset contains 51,583 referring descriptions of 11,046 objects" (Section 4.1 Dataset); "Cap3D Objaverse [45] dataset, is derived from Objaverse, one of the largest 3D datasets with around 800K objects. It features 660K 3D-text pairs" (Section 4.1 Dataset); "Text2Shape [12] contains 8,447 table instances and 6,591 chair instances from the ShapeNet dataset, along with 75,344 natural language descriptions." (Section 4.1 Dataset).
- Evidence for gains attributed to pretraining/multi-tasking: "As evidenced in Table 4, the pretraining stage significantly enhances performance across all downstream tasks." (Section 4.7 Ablation Study); "During pretraining, we simultaneously train the whole network with both object classification head, mask head, grounding head, text generation head, and text-3D matching head." (Section 3.4 Multi-Task Training).
- Evidence for gains attributed to model-size scaling: Not specified.

## 11. Architectural Workarounds
- Hierarchical 3D backbone: "The architecture of our point feature extraction network employs a sparse 3D convolutional U-net structure based on the MinkowskiEngine framework [19], featuring both an encoder and a decoder network." (Section 3.2 Point Cloud and Text Encoder).
- Voxelization (fixed grid assumption): "A colored input point cloud, denoted as  $\\mathbf{P} \\in \\mathbb{R}^{N_0 \\times 6}$ , undergoes quantization into  $N_0$  voxels represented as  $\\mathbf{V}_0 \\in \\mathbb{R}^{N_0 \\times 3}$ , with each voxel capturing the average RGB color from the points it contains as the initial voxel features." (Section 3.2 Point Cloud and Text Encoder).
- Masked attention to limit attention scope: "use masked attention instead of vanilla cross-attention where each query only attends to masked voxels predicted by the previous layer." (Section 3.3 Query Transformer Module).
- Voxel sampling for fixed-length inputs: "Current transformer implementations generally require a fixed length of inputs in each batch entry. To enable efficient batch-wise training" (Section 3.3 Query Transformer Module).
- Task router with functional heads: "To support diverse 3D vision-language tasks, we design multiple functional heads thus different tasks can be achieved by compositions of heads." (Section 3.4 Task Router).

## 12. Explicit Limitations and Non-Claims
- Limitation/future work: "To leverage the benefits of both point-based and projection-based techniques, our future work will focus on a hybrid approach. This strategy aims to concurrently learn joint 2D and 3D features, integrating insights from 2D foundation models." (Section D. Limitation and Future Work)
- Context for limitation: "This approach marks a departure from conventional 3D vision-language models that predominantly rely on projected multi-view images." (Section D. Limitation and Future Work)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)
**Constraint Profile:**
- Domain scope: Multiple 3D datasets (indoor scans and object shapes) plus language, but still confined to 3D point clouds with text.
- Task structure: Many predefined tasks (segmentation, detection, grounding, captioning, retrieval, classification) routed via explicit task heads.
- Representation rigidity: Voxelized point clouds with fixed voxel sizes, fixed-length sampling for batching, and dataset-specific cropping/minimum points.
- Model sharing vs specialization: Shared parameters across tasks with a router and heads, followed by task-specific finetuning.
- Role of positional encoding: Unspecified in the provided text.

### 14. Final Classification
**Classification:** Multi-task, multi-domain (constrained)
The paper evaluates multiple tasks such as "semantic segmentation, object detection, instance segmentation, visual grounding, 3D captioning, and text-3D cross-modal retrieval" (Abstract), plus "zero-shot 3D classification" on ModelNet datasets (Section B.1 Zero-Shot 3D Classification). The evaluation spans multiple 3D domains and datasets (ScanNet/ScanRefer/S3DIS for scans and Cap3D Objaverse/Text2Shape/ModelNet for shapes) (Section 4.1 Dataset), yet remains confined to 3D point clouds paired with language (Abstract; Section 3.1 Method overview), indicating a constrained multi-domain setup rather than unrestrained learning.
