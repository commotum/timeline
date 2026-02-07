# POS-BERT: Point Cloud One-Stage BERT Pre-Training (2022)
Source: POS-BERT- Point cloud one-stage BERT pre-training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reconstruction (masked patch modeling) | point cloud patches (masked tokens) | 3D (x, y, z) | Fixed | Static (inferred) | Direct (inferred) | masked patch representations | 3D (x, y, z) | Fixed |
| contrastive representation learning | augmented point clouds (global/local crops) | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Direct (inferred) | class token representations | 0D | Fixed |
| classification | point clouds | 3D (x, y, z) | Fixed | Static (inferred) | Direct (inferred) | object class label (inferred) | 0D | Fixed (inferred) |
| segmentation (part segmentation) | point clouds | 3D (x, y, z) | Fixed | Static (inferred) | Direct (inferred) | per-point part labels | 3D (x, y, z) | Fixed |

## Summary
The paper covers self-supervised point cloud pre-training via masked patch modeling and contrastive class-token consistency, plus downstream 3D object classification and part segmentation. Inputs are 3D point clouds, with outputs ranging from global 0D class-token/label outputs to per-point 3D segmentation labels. Fixed-size point sampling is described for the reported datasets, while attention and state properties are inferred from the standard transformer encoder description. Overall, the task domains stay within 3D point clouds and emphasize static-attention, direct-state processing.

## Evidence
### Task: reconstruction (masked patch modeling)
- "we use the mask patch modeling (MPM) task to perform point cloud pre-training, which aims to recover masked patches information" (Abstract)
- "we also use a mask patch modeling task to pretrain the point cloud Transformer." (Section 3.2)
- "divide 2048 points into 64 groups, where each group contains 32 points." (Section 4.2 Dataset)
- "the raw point clouds  $P \in \mathbb{R}^{N \times 3}$" (Section 3 Method)
- "We used a standard transformer as the Encoder backbone, which consists of a series of stacked multihead self-attention layers" (Section 3.1)
- Inference: Attention Dynamic = Static (inferred) and State Dynamic = Direct (inferred) based on the standard transformer encoder with multihead self-attention and no described external state.

### Task: contrastive representation learning
- "we combine contrastive learning to maximize the class token consistency between different transformation point clouds." (Abstract)
- "the global feature loss loss  $\mathcal{L}_{GFC}$  between the Encoder outputs' class token and the Momentum Encoder outputs' class token." (Section 3 Method)
- "global point cloud set  $P_g$  and the local point cloud set  $P_l$  are obtained by cropping the raw point clouds" (Section 3 Method)
- "the raw point clouds  $P \in \mathbb{R}^{N \times 3}$" (Section 3 Method)
- "We used a standard transformer as the Encoder backbone, which consists of a series of stacked multihead self-attention layers" (Section 3.1)
- Inference: Attention Dynamic = Static (inferred) and State Dynamic = Direct (inferred) based on the standard transformer encoder with multihead self-attention and no described external state.

### Task: classification
- "Linear SVM classification task has become a classic task to evaluate self-supervised point cloud representation learning." (Section 5.1)
- "We first performed fine-tuning experiments on point cloud classification tasks using a pretraining model." (Section 5.2)
- "We follow Yu et al. to sample 8192 points from each CAD model surface." (Section 4.2 Dataset)
- "We use a fully connected MLP network that combines ReLU, BN, and Dropout operations as the classification head." (Section 4.1)
- "We used a standard transformer as the Encoder backbone, which consists of a series of stacked multihead self-attention layers" (Section 3.1)
- Inference: Output is an object class label (inferred) with fixed-size output (inferred) based on the classification head, and Attention Dynamic = Static (inferred) plus State Dynamic = Direct (inferred) based on the standard transformer encoder description.

### Task: segmentation (part segmentation)
- "Compared with the classification task, the segmentation task needs to obtain the label of each point intensively." (Section 5.2)
- "the segmentation task needs to predict pre-point labels." (Section 4.1)
- "we randomly select 2048 points as input." (Section 4.2 Dataset)
- "We used a standard transformer as the Encoder backbone, which consists of a series of stacked multihead self-attention layers" (Section 3.1)
- Inference: Attention Dynamic = Static (inferred) and State Dynamic = Direct (inferred) based on the standard transformer encoder with multihead self-attention and no described external state.

## CSV Output (required)
CSV file written to `/home/jake/Developer/timeline/BIBLIOTHEQUE/02_XFORM-DIM/POS-BERT- Point cloud one-stage BERT pre-training/.TASK-DOMAINS.csv.tmp.5900564cf40345a1be2d51e128bc5748`.
