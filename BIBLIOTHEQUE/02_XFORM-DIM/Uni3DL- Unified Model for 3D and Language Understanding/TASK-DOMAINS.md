# Uni3DL: Unified Model for 3D and Language Understanding (Year not specified in the paper.)
Source: Uni3DL- Unified Model for 3D and Language Understanding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic segmentation | point clouds | 3D (x, y, z) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | semantic segmentation masks / per-point semantic labels | 3D (x, y, z) (inferred) | Capped (inferred) |
| instance segmentation | point clouds | 3D (x, y, z) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | instance masks and object classes | 3D (x, y, z); 0D (inferred) | Capped (inferred) |
| object detection | point clouds | 3D (x, y, z) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | 3D object bounding boxes and classes | 3D (x, y, z); 0D (inferred) | Capped (inferred) |
| grounded segmentation (visual grounding) | point clouds; referring text descriptions | 3D (x, y, z); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | grounded object masks | 3D (x, y, z) (inferred) | Capped (inferred) |
| 3D captioning | point clouds | 3D (x, y, z) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | caption tokens | 1D (t) (inferred) | Capped (inferred) |
| text-to-3D retrieval | text queries; 3D shapes | 1D (t); 3D (x, y, z) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | ranked 3D shapes / text-3D matches | 3D (x, y, z) (inferred) | Capped (inferred) |
| 3D object classification (zero-shot shape classification) | point clouds | 3D (x, y, z) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | class label | 0D (inferred) | Fixed (inferred) |
| grounded localization | point clouds; referring text descriptions | 3D (x, y, z); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | grounded object bounding boxes | 3D (x, y, z) (inferred) | Capped (inferred) |

## Summary
The paper presents a single unified 3D vision-language model that explicitly covers eight tasks: semantic segmentation, instance segmentation, object detection, grounded segmentation, captioning, text-to-3D retrieval, 3D object classification, and grounded localization. Inputs span 3D point clouds and language, and outputs span masks, boxes, class labels, retrieved 3D items, and generated text, so the justified range covers 0D, 1D (t), and 3D (x, y, z). Based on the OCR architectural description, task interfaces are mostly Capped (fixed-length batching/sampling and top-K style outputs), attention is Dynamic (masked attention conditioned on prior predictions), and state is Constructed (latent/text queries and learned semantic/mask representations).

## Evidence
### Task: semantic segmentation
- "With a unified architecture, Uni3DL supports diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, grounded segmentation, captioning, text-3D cross-modality retrieval, (zero-shot) 3D object classification." (Figure 1 caption)
- "For model evaluation, other than ScanNet (v2), ScanRefer, Cap3D, we use additional S3DIS [2] to evaluate both semantic and instance segmentation, Text2Shape [12] to evaluate text-to-3D retrieval." (Section 4.1. Dataset)
- Inference: Input/Output dimensions are marked 3D (x, y, z) (inferred) because the model is fed an "input point cloud" and predicts masks over voxelized point features; In/Out Dynamics are marked Capped (inferred) because "Current transformer implementations generally require a fixed length of inputs in each batch entry" (Section 3.3. Query Transformer Module); Attention Dynamic is marked Dynamic (inferred) because "each query only attends to masked voxels predicted by the previous layer" (Section 3.3. Query Transformer Module); State Dynamic is marked Constructed (inferred) because the model uses "learnable latent queries" and decoder-produced semantic/mask outputs (Section 3.1).

### Task: instance segmentation
- "the 3D instance segmentation task includes two heads, object classification, and mask prediction." (Section 3.4. Task Router)
- "We pretrain our Uni3DL on three datasets, including Scan-Net (v2) [20] for instance segmentation, ScanRefer [9] for visual grounding, and Cap3D Objaverse [45] dataset for 3D captioning and text-3D cross-modal retrieval." (Section 4.1. Dataset)
- Inference: Out Dimension includes 3D (x, y, z); 0D (inferred) because instance segmentation combines "object classification" (label output) and "mask prediction" (spatial output) (Section 3.4); Out Dynamics is Capped (inferred) because "the top 200 (for S3DIS) and 500 (for ScanNet (v2)) instances with the highest classification scores are retained" (Section 4.2. Implementation Details); Attention/State and input-side Dimension/Dynamics follow the same architecture evidence as above (Sections 3.1 and 3.3).

### Task: object detection
- "With a unified architecture, Uni3DL supports diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, grounded segmentation, captioning, text-3D cross-modality retrieval, (zero-shot) 3D object classification." (Figure 1 caption)
- "We compare 3D semantic segmentation, object detection, and instance segmentation performance with previous STOA methods in Table 3." (Section 4.3. 3D Semantic/Instance Sementation)
- Inference: Out Dimension is 3D (x, y, z); 0D (inferred) because detection outputs boxes plus classes by task intent; Dynamics are Capped (inferred) based on fixed-length batching in the shared decoder and top-scored retained predictions in related detection/instance reporting (Sections 3.3 and 4.2); Attention is Dynamic and State is Constructed (inferred) from masked attention and latent-query decoding (Sections 3.1 and 3.3).

### Task: grounded segmentation (visual grounding)
- "With a unified architecture, Uni3DL supports diverse 3D vision-language understanding tasks, including semantic segmentation, object detection, instance segmentation, grounded segmentation, captioning, text-3D cross-modality retrieval, (zero-shot) 3D object classification." (Figure 1 caption)
- "Visual grounding requires matching text descriptions to visual objects." (Section 3.4. Task Router, Grounding Head)
- Inference: In Dimension is 3D (x, y, z); 1D (t) (inferred) because grounding consumes point clouds plus referring text; Out Dimension is 3D (x, y, z) (inferred) because output is grounded masks; Dynamics are Capped (inferred) from fixed-length voxel handling (Section 3.3); Attention is Dynamic (masked attention) and State is Constructed (latent/text queries and semantic/mask outputs) (Sections 3.1 and 3.3).

### Task: 3D captioning
- "The Uni3DL is a versatile architecture tailored for diverse 3D vision-language tasks, including 3D object classification, text-to-3D retrieval, 3D captioning, 3D semantic and instance segmentation, and 3D visual grounding." (Section 3.1. Method overview)
- "During inference, our model predicts one token at each time and gets 3D captions in an autoregressive way." (Section 3.4. Text Generation Head)
- Inference: Out Dimension is 1D (t) (inferred) because output is a token sequence; Out Dynamics is Capped (inferred) because caption logits are defined over "the last  L_T semantic outputs" with matrix "S_cap in R^{L_T x V}" (Section 3.4); Attention is Dynamic (masked attention in decoder) and State is Constructed (latent/semantic query representations) (Sections 3.1 and 3.3).

### Task: text-to-3D retrieval
- "The Uni3DL is a versatile architecture tailored for diverse 3D vision-language tasks, including 3D object classification, text-to-3D retrieval, 3D captioning, 3D semantic and instance segmentation, and 3D visual grounding." (Section 3.1. Method overview)
- "Given a batch of B text-shape pairs, the retrieval head computes the similarities between 3D shape embeddings and corresponding text embeddings as  $\mathbf{S}_{ret} \in \mathbb{R}^{B \times B}$ , and calculates retrieval loss as:" (Section 3.4. Text-3D Matching Head)
- Inference: In Dimension is 1D (t); 3D (x, y, z) (inferred) because retrieval matches text and 3D shapes; Out Dimension is 3D (x, y, z) (inferred) for text-to-3D ranked outputs; Dynamics are Capped (inferred) due finite batch/candidate comparisons and reported R@K retrieval; Attention is Dynamic and State is Constructed (shared decoder/query representations) (Sections 3.1, 3.3, and 4.6).

### Task: 3D object classification (zero-shot shape classification)
- "The Uni3DL is a versatile architecture tailored for diverse 3D vision-language tasks, including 3D object classification, text-to-3D retrieval, 3D captioning, 3D semantic and instance segmentation, and 3D visual grounding." (Section 3.1. Method overview)
- "We use our Uni3DL model fine-tuned on the Cap3D Objaverse dataset to evaluate zero-shot 3D classification performance on ModelNet40 and ModelNet10 datasets." (Section B.1. Zero-Shot 3D Classification)
- Inference: Out Dimension is 0D (inferred) and Out Dynamics is Fixed (inferred) because classification returns class decisions (top-1/top-5 accuracies reported in Section B.1); input-side Dimension/Dynamics and Attention/State are inferred from point-cloud input with fixed-length batching plus latent-query masked-attention decoding (Sections 3.1 and 3.3).

### Task: grounded localization
- "Previous methods have also explored the grounded localization task." (Section B.2. Grounded Localization)
- "To produce grounded object location, we directly use grounded object masks to calculate their bounding boxes." (Section B.2. Grounded Localization)
- Inference: This is treated as a distinct localization task because the paper separately reports "grounded localization" (Section B.2); In Dimension is 3D (x, y, z); 1D (t) (inferred) from point cloud + referring text, Out Dimension is 3D (x, y, z) (inferred) for boxes, and Dynamics are Capped (inferred) due fixed-length shared processing; Attention is Dynamic and State is Constructed (inferred) from the same masked-attention latent-query pipeline used to form grounded masks before box conversion (Sections 3.1 and 3.3).
