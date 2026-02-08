# Transformer-based Point Cloud Generation Network (2023)
Source: Transformer-based Point Cloud Generation Network.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D point cloud generation | latent vectors | 1D (t) (inferred) | Fixed | Dynamic (inferred) | Constructed (inferred) | point clouds (2048-point sets of 3D coordinates) | 3D (x, y, z) | Fixed |
| point cloud classification | point clouds (processed by trained discriminator features) | 3D (x, y, z) | Fixed | Static (inferred) | Constructed (inferred) | object class labels (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers two task intents in the same 3D point-cloud modality: generation as the core task and classification as an auxiliary evaluation task. The generator maps a fixed-size latent vector to fixed-size 3D point clouds, while the classification setup maps point clouds to class labels. The supported dimensions therefore span inferred 1D latent inputs, 3D spatial point-cloud inputs/outputs, and inferred 0D label outputs. Dynamics are fixed by the described interface sizes, with inferred dynamic attention for generation (k-NN plus attention maps) and inferred static attention for linear-SVM classification.

## Evidence
### Task: 3D point cloud generation
- "In this paper, we propose a novel transformer-based 3D point cloud generation network to generate realistic point clouds." (Section ABSTRACT)
- "Our model aims to generate high-quality point clouds from a latent vector input." (Section 3.1.1 Overall architecture)
- "Our model generates a point cloud with 2048 points from a 128-dimensional latent vector." (Section 4.1.2 Implementation details)
- Inference: `In Dimension = 1D (t)` is inferred from the explicit "128-dimensional latent vector"; `Attention Dynamic = Dynamic` is inferred from runtime neighborhood/attention construction ("we first employ the k-NN operation to construct three neighborhoods of different scales for each point" in Section 3.1.2 and "the attention map adaptively learns the correlation between the neighbor points in feature and coordinate space" in Section 3.1.3); `State Dynamic = Constructed` is inferred because the model constructs intermediate states such as initial, upsampled, and refined point features before regressing final coordinates (Section 3.1.1 to Section 3.1.3).

### Task: point cloud classification
- "To investigate the representation learning ability of the model, we conduct classification experiments as in previous methods [11, 36, 39]." (Section 4.2.3 Classification results)
- "with all the data of ShapeNet, then use the trained discriminator to extract features to train a linear SVM for classification on ModelNet10 and ModelNet40." (Section 4.2.3 Classification results)
- "Table 2: Classification accuracy of various methods on ModelNet10 (MN10) and ModelNet40 (MN40) datasets." (Section 4.2.3 Classification results)
- Inference: `Attention Dynamic = Static` is inferred because the described classifier is a linear SVM on extracted features with no runtime input-selection mechanism specified; `State Dynamic = Constructed` is inferred from "extract features" using the trained discriminator; `Output = object class labels`, `Out Dimension = 0D`, and `Out Dynamics = Fixed` are inferred from the explicit classification intent and fixed-label prediction setup in Section 4.2.3.
