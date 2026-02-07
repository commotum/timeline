# Masked Autoencoders for Point Cloud Self-supervised Learning (Not specified in the paper)
Source: Masked Autoencoders for Point Cloud Self-supervised Learning (Point-MAE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked point patch reconstruction | masked point cloud patches | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | reconstructed point patches (coordinates) | 3D (x, y, z) (inferred) | Capped (inferred) |
| Object classification | 3D point clouds (objects) | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | object class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Few-shot object classification | 3D point clouds (objects) | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | object class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Part segmentation | 3D point clouds (objects) | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | per-point part labels | 3D (x, y, z) (inferred) | Capped (inferred) |

## Summary
The paper defines a self-supervised reconstruction task for masked point patches and evaluates downstream 3D point cloud tasks: object classification (including few-shot settings) and part segmentation. Inputs are 3D point clouds, while outputs are either reconstructed 3D point patches, per-point part labels, or single class labels. Based on the described Transformer autoencoder and point sampling/patching scheme, attention is static and state is direct, with capped input/output sizes tied to point counts.

## Evidence
### Task: Masked point patch reconstruction
- "learns high-level latent features from unmasked point patches, aiming to reconstruct the masked point patches." (Abstract)
- "Our reconstruction target is to recover coordinates of the points in every masked point patch." (Section 3.3 Reconstruction Target)
- "point cloud consists of unordered points in 3D space." (Section 3.1 Point Cloud Masking and Embedding)
- "for different resolutions of the input point cloud, we divide them into different numbers of patches with a linear scaling." (Section 4 Experiments)
- "Our encoder consists of standard Transformer blocks and only encodes visible tokens  $T_v$  without mask tokens  $T_m$ ." (Section 3.2 Autoencoder's Backbone)
- "Transformers [40] model global dependencies of input through the self-attention mechanism" (Section 2.3 Transformers)
- Inference: In/Out Dimension and In/Out Dynamics are inferred from the 3D coordinate definition and variable patch counts; Attention is Static and State is Direct based on the standard Transformer encoder-decoder that processes the full token sequence without any described external or persistent state.

### Task: Object classification
- "we evaluate our pre-trained model on a challenging real-world dataset, ScanObjectNN [39]" (Section 4.2 Downstream Tasks)
- "We evaluate our pre-trained model on ModelNet40 [46] for object classification." (Section 4.2 Downstream Tasks)
- "all the reported methods are given 1024 points that only contain coordinate information" (Section 4.2 Downstream Tasks)
- "point cloud consists of unordered points in 3D space." (Section 3.1 Point Cloud Masking and Embedding)
- "for different resolutions of the input point cloud, we divide them into different numbers of patches with a linear scaling." (Section 4 Experiments)
- "Our encoder consists of standard Transformer blocks and only encodes visible tokens  $T_v$  without mask tokens  $T_m$ ." (Section 3.2 Autoencoder's Backbone)
- "Transformers [40] model global dependencies of input through the self-attention mechanism" (Section 2.3 Transformers)
- Inference: Input/Output Dimensions and Dynamics, Attention, and State are inferred from the 3D point-cloud definition, variable patch counts, and standard Transformer encoder-decoder; Output is a 0D class label inferred from the "object classification" task framing.

### Task: Few-shot object classification
- "Few-shot Learning We follow previous works [54,37,41] to conduct few-shot learning experiments on ModelNet40 [46]" (Section 4.2 Downstream Tasks)
- "Table 3. Few-shot object classification on ModelNet40." (Section 4.2 Downstream Tasks)
- "point cloud consists of unordered points in 3D space." (Section 3.1 Point Cloud Masking and Embedding)
- "for different resolutions of the input point cloud, we divide them into different numbers of patches with a linear scaling." (Section 4 Experiments)
- "Our encoder consists of standard Transformer blocks and only encodes visible tokens  $T_v$  without mask tokens  $T_m$ ." (Section 3.2 Autoencoder's Backbone)
- "Transformers [40] model global dependencies of input through the self-attention mechanism" (Section 2.3 Transformers)
- Inference: Input/Output Dimensions and Dynamics, Attention, and State are inferred from the 3D point-cloud definition, variable patch counts, and standard Transformer encoder-decoder; Output is a 0D class label inferred from the few-shot object classification framing.

### Task: Part segmentation
- "Part Segmentation We evaluate the representation learning capability of our Point-MAE on ShapeNetPart dataset [53]" (Section 4.2 Downstream Tasks)
- "We follow previous works [29,30,54] to sample 2048 points as input for each object" (Section 4.2 Downstream Tasks)
- "MLP is adopted to predict the label for each point." (Section 4.2 Downstream Tasks)
- "point cloud consists of unordered points in 3D space." (Section 3.1 Point Cloud Masking and Embedding)
- "for different resolutions of the input point cloud, we divide them into different numbers of patches with a linear scaling." (Section 4 Experiments)
- "Our encoder consists of standard Transformer blocks and only encodes visible tokens  $T_v$  without mask tokens  $T_m$ ." (Section 3.2 Autoencoder's Backbone)
- "Transformers [40] model global dependencies of input through the self-attention mechanism" (Section 2.3 Transformers)
- Inference: In/Out Dimensions and Dynamics, Attention, and State are inferred from the 3D point-cloud definition, variable patch counts, and standard Transformer encoder-decoder; the output is a 3D-indexed label field because labels are predicted for each input point.
