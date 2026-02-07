# OneFormer3D: One Transformer for Unified Point Cloud Segmentation (Not specified in the paper.)
Source: OneFormer3D- One Transformer for Unified Point Cloud Segmentation (OneFormer3D).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D semantic segmentation | 3D point cloud (points with RGB+XYZ) | 3D (x, y, z) | Open (inferred) | Static (inferred) | Constructed (inferred) | Semantic masks (per-point semantic labels) | 3D (x, y, z) | Open (inferred) |
| 3D instance segmentation | 3D point cloud (points with RGB+XYZ) | 3D (x, y, z) | Open (inferred) | Static (inferred) | Constructed (inferred) | Instance masks for individual objects (subset of points) | 3D (x, y, z) | Open (inferred) |
| 3D panoptic segmentation | 3D point cloud (points with RGB+XYZ) | 3D (x, y, z) | Open (inferred) | Static (inferred) | Constructed (inferred) | Foreground instance masks plus semantic labels for background points | 3D (x, y, z) | Open (inferred) |
| 3D object detection | 3D point cloud (points with RGB+XYZ) | 3D (x, y, z) | Open (inferred) | Static (inferred) | Constructed (inferred) | Axis-aligned 3D bounding boxes (from predicted instances) | 3D (x, y, z) | Capped (inferred) |

## Summary
OneFormer3D addresses 3D semantic, instance, and panoptic segmentation on 3D point clouds, and is additionally adapted to 3D object detection by boxing predicted instances. Across tasks, inputs and outputs are spatial 3D (x, y, z) point-cloud structures; input size is treated as Open (inferred) because the paper describes variable N-point clouds without a fixed cap. The decoder uses self- and cross-attention over provided superpoint features and constructs kernels from queries, so attention is Static (inferred) and state is Constructed (inferred). For detection, output is a capped set of 3D boxes derived from a fixed number of instance queries (inferred).

## Evidence
### Task: 3D semantic segmentation
- "Taking a 3D point cloud as input, our trained model solves 3D instance, 3D semantic, and 3D panoptic segmentation tasks." (Figure 2 caption)
- "Semantic segmentation outputs a mask for each semantic category, so that each point in a point cloud gets assigned with a semantic label." (Section 1. Introduction)
- "Then, superpoint features are convolved with these kernels to produce K_ins instance and K_sem semantic masks, respectively." (Section 3.2. Query Decoder)
- Inference: Dynamics marked Open because "Assuming that an input point cloud contains N points." Attention Static and State Constructed inferred from "self-attention on queries and cross-attention with keys and values from superpoint features" and "transforms them into K_ins + K_sem kernels." (Sections 3.1, 3.2)

### Task: 3D instance segmentation
- "Taking a 3D point cloud as input, our trained model solves 3D instance, 3D semantic, and 3D panoptic segmentation tasks." (Figure 2 caption)
- "Instance segmentation returns a set of masks of individual objects." (Section 1. Introduction)
- "Then, superpoint features are convolved with these kernels to produce K_ins instance and K_sem semantic masks, respectively." (Section 3.2. Query Decoder)
- Inference: Dynamics marked Open because "Assuming that an input point cloud contains N points." Attention Static and State Constructed inferred from "self-attention on queries and cross-attention with keys and values from superpoint features" and "transforms them into K_ins + K_sem kernels." (Sections 3.1, 3.2)

### Task: 3D panoptic segmentation
- "Taking a 3D point cloud as input, our trained model solves 3D instance, 3D semantic, and 3D panoptic segmentation tasks." (Figure 2 caption)
- "it implies predicting a mask for each foreground object (thing)" (Section 1. Introduction)
- "and a semantic label for each back-" (Section 1. Introduction)
- "ground point (stuff)." (Section 1. Introduction)
- "Panoptic prediction is obtained from instance and semantic outputs." (Section 3.4. Inference)
- Inference: Dynamics marked Open because "Assuming that an input point cloud contains N points." Attention Static and State Constructed inferred from "self-attention on queries and cross-attention with keys and values from superpoint features" and "transforms them into K_ins + K_sem kernels." (Sections 3.1, 3.2)

### Task: 3D object detection
- "Besides, we adopt OneFormer3D to 3D object detection by enclosing predicted 3D instances with tight axis-aligned 3D bounding boxes." (Section 4.2. Comparison to Prior Work)
- "Assuming that an input point cloud contains N points, the input can be formulated as P in R^{N x 6}." (Section 3.1. Backbone and Pooling)
- Inference: Input dynamics Open from "Assuming that an input point cloud contains N points." Output dynamics Capped because boxes come from predicted instances and the decoder uses "K_ins + K_sem queries as inputs." Attention Static and State Constructed inferred from "self-attention on queries and cross-attention with keys and values from superpoint features" and "transforms them into K_ins + K_sem kernels." (Sections 3.1, 3.2, 4.2)
