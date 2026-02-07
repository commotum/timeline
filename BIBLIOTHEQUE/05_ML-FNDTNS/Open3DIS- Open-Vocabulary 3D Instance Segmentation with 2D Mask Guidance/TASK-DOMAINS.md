# Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance (Not specified in the paper.)
Source: Open3DIS- Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open-vocabulary 3D instance segmentation | 3D point cloud; RGB-D sequence (RGB images, depth maps, camera matrices) | 3D (x, y, z); 3D (x, y, t) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | 3D binary instance masks (object instances) | 3D (x, y, z) | Capped (inferred) |
| Text-driven 3D instance segmentation | Text queries/prompts; 3D point cloud; RGB-D sequence (RGB images, depth maps, camera matrices) | 1D (t); 3D (x, y, z); 3D (x, y, t) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | 3D instance masks matched to text queries (similarity scores) | 3D (x, y, z); 0D | Capped (inferred) |

## Summary
Open3DIS targets open-vocabulary 3D instance segmentation from 3D point clouds and RGB-D sequences, and it also supports text-driven instance segmentation by matching 3D proposals to text prompts. The input modalities span 3D spatial point clouds and RGB-D video (3D (x, y, t)), with text queries adding a 1D sequence when used for querying. The paper supports capped-size inputs/outputs and dynamic selection of views (inferred), and it constructs intermediate proposal structures and features (constructed state, inferred).

## Evidence
### Task: Open-vocabulary 3D instance segmentation
- "This paper addresses the challenging problem of open-vocabulary 3D point cloud instance segmentation (OV-3DIS). Given a 3D scene represented by a point cloud, we seek to obtain a set of binary instance masks of any classes of interest, which may not exist during the training phase." (Section 1. Introduction)
- "This module takes as input a 3D point cloud  $\mathbf{P} = \{\mathbf{p}_n\}_{n=1}^N$ , where N is the number of points, and  $\mathbf{p}_i \in \mathbb{R}^6$  includes 3D coordinates and RGB color. Additionally, it receives an RGB-D video sequence  $\mathbf{V} = \{(\mathbf{I}_t, \mathbf{D}_t, \Pi_t)\}_{t=1}^T$ , where each frame t contains RGB image  $\mathbf{I}_t$ , depth map  $\mathbf{D}_t$ , and camera matrix  $\Pi_t$  (i.e., the product of intrinsic and extrinsic matrices used for projecting 3D points onto the image plane). The output comprises  $K_1$  binary instance masks represented in a  $K_1 \times N$  binary matrix  $\mathbf{M}_1$  (Fig. 2 - 3)." (Section 3.1)
- "In Pointwise Feature Extraction, each proposal is projected into all viewpoints, and we select the top  $\lambda$ =5 views with the largest number of projected points." (Section 4.1. Experimental Setup, Implementation Details)
- "In a pre-processing step, we utilize the method of [11] to group points into geometrically homogeneous regions, termed superpoints (Fig. 2 - 1)." (Section 3.1)
- "In the final stage of our pipeline, we compute a feature vector for each 3D object proposal from our combined proposal set." (Section 3.3)
- Inference: In/Out Dynamics are marked Capped (inferred) because the inputs/outputs are defined with finite counts N, T, and K in Section 3.1. Attention Dynamic is marked Dynamic (inferred) because the method selects the top-$\lambda$ views based on projected points (Section 4.1, Implementation Details). State Dynamic is marked Constructed (inferred) because the pipeline constructs superpoints and per-proposal feature vectors (Section 3.1; Section 3.3).

### Task: Text-driven 3D instance segmentation
- "Our approach processes a 3D point cloud and an RGB-D sequence, producing a set of 3D binary masks indicating object instances in the scene." (Section 3. Method)
- "In the final stage of our pipeline, we compute a feature vector for each 3D object proposal from our combined proposal set. This per-proposal feature vector serves various instance-based tasks, such as comparison with text prompts in the CLIP space [39]." (Section 3.3)
- "The final score between a text query  $\rho$  and a 3D mask  $\mathbf{m}_k^{\mathrm{3D}}$  is the average cosine similarity between its CLIP text embedding  $\mathbf{e}_{\rho}$  and all points within the mask, particularly:" (Section 3.3)
- "Our qualitative results with arbitrary text queries. We visualize the qualitative results of text-driven 3D instance segmentation in Fig. 5." (Section 4.2)
- "We query instance masks using arbitrary text prompts involving object categories that are not present in the ScanNet200 labels. For each scene, we showcase the instance that has the highest similarity score to the query's embedding." (Figure 5 caption)
- "In Pointwise Feature Extraction, each proposal is projected into all viewpoints, and we select the top  $\lambda$ =5 views with the largest number of projected points." (Section 4.1. Experimental Setup, Implementation Details)
- Inference: In/Out Dynamics are marked Capped (inferred) because the inputs/outputs are defined with finite counts N, T, and K in Section 3.1. Attention Dynamic is marked Dynamic (inferred) because the method selects the top-$\lambda$ views based on projected points (Section 4.1, Implementation Details). State Dynamic is marked Constructed (inferred) because the pipeline constructs per-proposal feature vectors for text comparison (Section 3.3).

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
"Open-vocabulary 3D instance segmentation","3D point cloud; RGB-D sequence (RGB images, depth maps, camera matrices)","3D (x, y, z); 3D (x, y, t)","Capped (inferred)","Dynamic (inferred)","Constructed (inferred)","3D binary instance masks (object instances)","3D (x, y, z)","Capped (inferred)"
"Text-driven 3D instance segmentation","Text queries/prompts; 3D point cloud; RGB-D sequence (RGB images, depth maps, camera matrices)","1D (t); 3D (x, y, z); 3D (x, y, t)","Capped (inferred)","Dynamic (inferred)","Constructed (inferred)","3D instance masks matched to text queries (similarity scores)","3D (x, y, z); 0D","Capped (inferred)"
