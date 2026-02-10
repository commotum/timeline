# Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks (2019)
Source: Set Transformer- A Framework for Attention-based Permutation-Invariant Neural Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maximum value regression | Set of real numbers `{x_1, ..., x_n}` | 0D (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Maximum value `max(x_1, ..., x_n)` | 0D (inferred) | Fixed (inferred) |
| Unique-character counting | Set of Omniglot character images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Number of different characters in the set | 0D (inferred) | Fixed (inferred) |
| Amortized clustering (mixture-of-Gaussians maximum likelihood) | Set of data points (synthetic 2D points or CIFAR-100 image feature vectors) | 2D (x, y) (inferred); 0D (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Mixture parameters `{π_j, μ_j, σ_j}` for `k` components | 2D (x, y) (inferred); 0D (inferred) | Fixed (inferred) |
| Set anomaly detection | Set of CelebA images (7 normal + 1 anomaly) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Anomalous image in the set | 0D (inferred) | Fixed (inferred) |
| Point cloud classification | 3D point cloud set (`n` vectors in `R^3`) | 3D (x, y, z) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Object category label (40 classes) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates Set Transformer on five set-input tasks: scalar regression, counting, amortized clustering/parameter estimation, anomaly detection, and object classification. The supported input domains span 0D sets of scalars, 2D image/planar data, and 3D point clouds, with mostly Capped input dynamics from explicit set-size ranges and one Fixed-size anomaly setup. Outputs are fixed-size scalar decisions/values or fixed-size clustering parameter sets. Across these tasks, the architecture uses attention over the provided set elements (Static, inferred under the glossary) and constructs latent representations/inducing summaries (Constructed, inferred).

## Evidence
### Task: Maximum value regression
- "To demonstrate the advantage of attention-based set aggregation over simple pooling operations, we consider a toy problem: regression to the maximum value of a given set." (Section 5.1)
- "Given a set of real numbers  $\{x_1, \ldots, x_n\}$ , the goal is to return  $\max(x_1, \dots, x_n)$ ." (Section 5.1)
- "We first sample a dataset size n uniformly from the set of integers  $\{1,\cdots,10\}$ ." (Section 2.1, Supplementary Material)
- Inference: `In Dimension = 0D`, `Out Dimension = 0D`, `In Dynamics = Capped`, and `Out Dynamics = Fixed` are inferred from scalar-set input, scalar output, and explicit bounded set size. `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from the shared Set Transformer architecture description: "a Set Transformer consists of an encoder followed by a decoder, but a distinguishing feature is that each layer in the encoder and decoder attends to their inputs" (Section 3).

### Task: Unique-character counting
- "we introduce a new task of counting unique elements in an input set." (Section 5.2)
- "we generate input sets by sampling between 6 and 10 images and we train the model to predict the number of different characters inside the set." (Section 5.2)
- "We used a Poisson regression model to predict this number" (Section 5.2)
- Inference: `In Dimension = 2D (x, y)` is inferred because the input elements are images; `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred because the target is one count per set; `In Dynamics = Capped` is inferred from the 6-10 image range. `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from Section 3's encoder-decoder attention design.

### Task: Amortized clustering (mixture-of-Gaussians maximum likelihood)
- "We applied the set-input networks to the task of maximum likelihood of mixture of Gaussians (MoGs)." (Section 5.3)
- "The goal is to learn the optimal parameters  $\theta^*(X) = \arg\max_{\theta}\log p(X;\theta)$ ." (Section 5.3)
- "we aim to learn a generic meta-algorithm that directly maps the input set X to  $\theta^*(X)$ ." (Section 5.3)
- "Synthetic 2D mixtures of Gaussians: Each dataset contains  $n \in [100, 500]$  points on a 2D plane" (Section 5.3)
- "**CIFAR-100**: Each dataset contains  $n \in [100, 500]$  images sampled from four random classes in the CIFAR-100 dataset. Each image is represented by a 512-dim vector" (Section 5.3)
- Inference: `In Dimension = 2D (x, y); 0D` and `Out Dimension = 2D (x, y); 0D` are inferred from the two stated input modalities (2D points vs. vector features) and parameter outputs `{π_j, μ_j, σ_j}`. `In Dynamics = Capped` is inferred from `n ∈ [100, 500]`; `Out Dynamics = Fixed` is inferred from fixed `k`-component parameterization in Eq. (18). `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from Section 3.

### Task: Set anomaly detection
- "We evaluate our methods on the task of meta-anomaly detection within a set using the CelebA dataset." (Section 5.4)
- "For every set, we select two attributes at random and construct the set by selecting seven images containing both attributes and one image with neither." (Section 5.4)
- "The goal of this task is to find the image that does not belong to the set." (Section 5.4)
- Inference: `In Dimension = 2D (x, y)` is inferred from image input; `In Dynamics = Fixed` is inferred from fixed 8-image set construction; `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from selecting one anomalous item per set. `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from Section 3.

### Task: Point cloud classification
- "We evaluated Set Transformers on a classification task using the ModelNet40 (Chang et al., 2015) dataset" (Section 5.5)
- "Each object is represented as a point cloud, which we treat as a set of n vectors in  $\mathbb{R}^3$ ." (Section 5.5)
- "We performed experiments with input sets of size  $n \in \{100, 1000, 5000\}$ ." (Section 5.5)
- "This dataset consists of a three-dimensional representation of 9,843 training and 2,468 test data which each belong to one of 40 object classes." (Section 2.5, Supplementary Material)
- Inference: `In Dimension = 3D (x, y, z)` is inferred from `R^3` point coordinates; `In Dynamics = Capped` is inferred from the enumerated set sizes; `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from one class decision per object. `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from Section 3.
