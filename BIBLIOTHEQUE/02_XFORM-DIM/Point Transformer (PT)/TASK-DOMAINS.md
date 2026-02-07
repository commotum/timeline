# Point Transformer (Not specified in the paper)
Source: Point Transformer (PT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic scene segmentation | 3D point cloud scenes | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | pointwise semantic labels | 3D (x, y, z) | Not specified in the paper. |
| 3D shape classification | 3D point sets sampled from CAD models with normals | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | object category label | 0D | Fixed |
| Object part segmentation | 3D point sets of objects | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | pointwise object part labels (inferred) | 3D (x, y, z) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates Point Transformer on three supervised 3D point-cloud tasks: semantic scene segmentation, object part segmentation, and 3D shape classification. Inputs are 3D point sets, with outputs either per-point semantic/part labels or a single object category label. The architecture applies local kNN self-attention, implying static attention and constructed state (inferred). The paper does not specify fixed or capped input/output sizes beyond describing point-set cardinality as N.

## Evidence
### Task: Semantic scene segmentation
- "For 3D semantic segmentation, we use the challenging Stanford Large-Scale 3D Indoor Spaces (S3DIS) dataset [1]." (Section 4. Experiments)
- "Each point in the scan is assigned a semantic label from 13 categories (ceiling, floor, table, etc.)." (Section 4.1. Semantic Segmentation)
- Inference: Attention is Static (inferred) because neighborhoods are fixed as "a local neighborhood (specifically, k nearest neighbors)" (Section 3.2. Point Transformer Layer). State is Constructed (inferred) because the block is "producing new feature vectors for all data points as its output." (Section 3.4. Point Transformer Block)

### Task: 3D shape classification
- "For 3D shape classification, we use the widely adopted ModelNet40 dataset [47]." (Section 4. Experiments)
- "uniformly sample the points from each CAD model together with the normal vectors from the object meshes." (Section 4.2. Shape Classification)
- "This global feature is passed through an MLP to get the global classification logits." (Section 3.5. Network Architecture, Output head)
- Inference: Attention is Static (inferred) because neighborhoods are fixed as "a local neighborhood (specifically, k nearest neighbors)" (Section 3.2. Point Transformer Layer). State is Constructed (inferred) because the block is "producing new feature vectors for all data points as its output." (Section 3.4. Point Transformer Block)

### Task: Object part segmentation
- "And for object part segmentation, we use ShapeNetPart [52]." (Section 4. Experiments)
- "The ShapeNetPart dataset [52] is annotated for 3D object part segmentation." (Section 4.3. Object Part Segmentation)
- "We use the sampled point sets produced by Qi et al. [27] for a fair comparison with prior work." (Section 4.3. Object Part Segmentation)
- "The number of parts for each category is between 2 and 6, with 50 different parts in total." (Section 4.3. Object Part Segmentation)
- Inference: Output is pointwise part labels and Out Dimension is 3D (x, y, z) (inferred) because the dataset is "annotated for 3D object part segmentation" and uses "sampled point sets" (Section 4.3. Object Part Segmentation). Attention is Static (inferred) because neighborhoods are fixed as "a local neighborhood (specifically, k nearest neighbors)" (Section 3.2. Point Transformer Layer). State is Constructed (inferred) because the block is "producing new feature vectors for all data points as its output." (Section 3.4. Point Transformer Block)
