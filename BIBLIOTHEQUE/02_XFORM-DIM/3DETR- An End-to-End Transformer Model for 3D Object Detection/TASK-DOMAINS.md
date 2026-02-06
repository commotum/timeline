# An End-to-End Transformer Model for 3D Object Detection (Not specified in the paper)
Source: 3DETR- An End-to-End Transformer Model for 3D Object Detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | 3D point cloud (unordered set of XYZ points) | 3D (x, y, z) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | 3D bounding boxes with class labels | 3D (x, y, z) (inferred) | Capped (inferred) |
| Shape classification | 3D point cloud with normals (sampled point set) | 3D (x, y, z) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Shape class label distribution | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers two tasks on 3D point-cloud inputs: 3D object detection and shape classification. Detection outputs a set of 3D bounding boxes with class labels, while classification outputs a distribution over predefined shape classes. The inputs are fixed-size sampled point sets, and the detection outputs are capped by a fixed number of queries; attention is standard self-/cross-attention over those fixed sets and the model operates as a direct feedforward mapping from inputs to outputs (inferred from the architecture description).

## Evidence
### Task: 3D object detection
- "We propose 3DETR, an end-to-end Transformer based object detection model for 3D point clouds." (Abstract)
- "3DETR takes as input a 3D point cloud and predicts the positions of objects in the form of 3D bounding boxes." (Section 3.2)
- "A point cloud is a unordered set of N points where each point is associated with its 3-dimensional XYZ coordinates." (Section 3.2)
- "This decoder takes as input the N' point features and a set of B query embeddings  $\{\mathbf{q}_1^e,\ldots,\mathbf{q}_B^e\}$  to produce a set of B features that are then used to predict 3D-bounding boxes." (Section 3.2)
- "The Transformer encoder produces a set of per-point features using multiple layers of self-attention." (Figure 2)
- "We use a single set aggregation operation [45] to subsample N'=2048 points and obtain 256 dimensional point features." (Section 3.5)
- Inference: In Dimension and Out Dimension are 3D (x, y, z) because inputs are XYZ points and outputs are 3D bounding boxes; In Dynamics is Fixed because inputs are subsampled to N'=2048 points; Out Dynamics is Capped because the decoder predicts a fixed set of B boxes; Attention Dynamic is Static because the encoder/decoder apply standard self-attention over a fixed set of points; State Dynamic is Direct because the model is described as an end-to-end mapping from the point cloud to boxes without an external memory state. (Sections 3.2, 3.5)

### Task: Shape classification
- "We report shape classification results by training our Transformer encoder model." (Table 4 / Section 4.2.1)
- "To verify that our encoder design is not specific to the detection task we test the encoder on shape classification of of models including 3D Warehouse [79]." (Section 4.2.1)
- "We use the three layer encoder from 3DETR with vanilla self-attention (no decoder) or the three layer encoder from 3DETR-m." (Section 4.2.1)
- "We use the processed point clouds with normals from [45], and sample 8192 points as input for both training and testing our models." (Section B.7)
- "Architecture Details. We use the base 3DETR and 3DETR-m encoder architectures, followed by a 2-layer MLP with batch norm and a 0.5 dropout to transform the final features into a distribution over the 40 predefined shape classes." (Section B.7)
- Inference: In Dimension is 3D (x, y, z) because inputs are 3D point clouds with normals; In Dynamics is Fixed because inputs are sampled to 8192 points; Out Dimension is 0D because the output is a class distribution; Attention Dynamic is Static because the encoder uses vanilla self-attention over the fixed point set; State Dynamic is Direct because the task is framed as an encoder-to-MLP feedforward mapping without persistent memory. (Sections 4.2.1, B.7)
