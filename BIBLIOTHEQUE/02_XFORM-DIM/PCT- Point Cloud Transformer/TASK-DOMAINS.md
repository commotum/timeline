# PCT: Point Cloud Transformer (Not specified in the paper)
Source: PCT- Point Cloud Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shape classification | point cloud (3D points) | 3D (x, y, z) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class label (object category) | 0D | Fixed (inferred) |
| Part segmentation | point cloud (3D points) | 3D (x, y, z) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | point-wise part labels | 3D (x, y, z) | Fixed (inferred) |
| Semantic segmentation | point cloud (3D points) | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | point-wise semantic labels | 3D (x, y, z) | Not specified in the paper. |
| Normal estimation | point cloud (3D points) | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | point-wise normals | 3D (x, y, z) | Not specified in the paper. |

## Summary
PCT is evaluated on four point-cloud tasks: shape classification, part segmentation, semantic segmentation, and normal estimation, all using 3D point clouds as inputs. Outputs are either a single object class label or point-wise predictions (part labels, semantic labels, or normals) over the same 3D point set. The paper fixes point counts in the classification and part-segmentation experiments (1,024 or 2,048 points), while input/output dynamics for semantic segmentation and normal estimation are not specified. The attention mechanism uses global self-attention over all points (Static, inferred) and the model constructs latent point features for downstream decisions (Constructed, inferred).

## Evidence
### Task: Shape classification
- "To classify a point cloud  $\mathcal{P}$  into  $N_c$  object categories (e.g. desk, table, chair)" (Section 3.1, Classification)
- "predict the final classification scores  $\mathcal{C} \in \mathbb{R}^{N_c}$ ." (Section 3.1, Classification)
- Inference: Marked In Dynamics as Fixed (inferred) because inputs were "uniformly sample each object to 1,024 points." (Section 4.1). Marked Attention Dynamic as Static (inferred) because attention is "related to all input features" and uses "global context." (Section 1). Marked State Dynamic as Constructed (inferred) because PCT "transform (encode) the input points into a new higher dimensional feature space," (Section 3.1). Marked Out Dynamics as Fixed (inferred) because output is "classification scores  $\mathcal{C} \in \mathbb{R}^{N_c}$ ." (Section 3.1).

### Task: Part segmentation
- "For the task of segmenting the point cloud into  $N_s$  parts (e.g. table top, table legs; a part need not be contiguous)" (Section 3.1, Segmentation)
- "predict the final point-wise segmentation scores  $\mathcal{S} \in \mathbb{R}^{N \times N_s}$" (Section 3.1, Segmentation)
- Inference: Marked In Dynamics as Fixed (inferred) because "all models were downsampled to 2,048 points" in the part-segmentation setup. (Section 4.3). Marked Attention Dynamic as Static (inferred) because attention is "related to all input features" and uses "global context." (Section 1). Marked State Dynamic as Constructed (inferred) because PCT "transform (encode) the input points into a new higher dimensional feature space," (Section 3.1). Marked Out Dynamics as Fixed (inferred) because output is "segmentation scores  $\mathcal{S} \in \mathbb{R}^{N \times N_s}$" and the dataset has "50 part labels;" (Section 3.1; Section 4.3).

### Task: Semantic segmentation
- "The S3DIS is a indoor scene dataset for point cloud semantic segmentation." (Section 4.4)
- "Each point in the dataset is divided into 13 categories." (Section 4.4)
- Inference: Marked Attention Dynamic as Static (inferred) because attention is "related to all input features" and uses "global context." (Section 1). Marked State Dynamic as Constructed (inferred) because PCT "transform (encode) the input points into a new higher dimensional feature space," (Section 3.1).

### Task: Normal estimation
- "The surface normal estimation is to determine the normal direction at each point." (Section 4.2)
- "For the task of normal estimation, we use the same architecture as in segmentation by setting  $N_s=3$" (Section 3.1, Normal estimation)
- Inference: Marked Attention Dynamic as Static (inferred) because attention is "related to all input features" and uses "global context." (Section 1). Marked State Dynamic as Constructed (inferred) because PCT "transform (encode) the input points into a new higher dimensional feature space," (Section 3.1).
