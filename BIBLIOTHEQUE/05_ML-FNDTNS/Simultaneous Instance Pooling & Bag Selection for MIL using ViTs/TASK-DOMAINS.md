# Simultaneous instance pooling and bag representation selection approach for multiple-instance learning (MIL) using vision transformer (2024)
Source: Simultaneous Instance Pooling & Bag Selection for MIL using ViTs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bag-level classification (binary and multi-class MIL) | Bags of instances (molecular conformations; image segments/patches; images) | 0D; 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed | Bag label (class decision) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper focuses on a single core task: bag-level classification in multiple-instance learning, covering both binary and multi-class settings. Inputs are bags with variable numbers of instances, spanning vectorized molecular conformations and 2D image instances (segments, patches, or whole images). The method constructs latent instance and bag representations, and the RSN selects among candidate bag representations at runtime. Under the glossary mapping, this supports Capped input dynamics, Dynamic attention behavior, and 0D class-label outputs.

## Evidence
### Task: Bag-level classification (binary and multi-class MIL)
- "In this paper, we concentrate on bag-level classification for binary and multi-class MIL applications." (Section 3.1 Problem formulation)
- "Therefore, a representation vector is generated for the bag of instances and the model classifies the bag representation vector instead of individual instances." (Section 3.1 Problem formulation)
- "In binary MIL classification problem, for a given bag  $B_i = \{x_{i,1}, x_{i,2}, x_{i,3}, \dots, x_{i,mi}\}$  of mi total instances with d dimensions" (Section 3.1 Problem formulation)
- "This process transforms the bag with a variable number of instances to a manageable vector representation" (Section 3.5 Computation of bag representation vectors)
- "RSN aims to select one of the representation vectors, which is most informative for the bag classification." (Section 3.6 Representation selection subnetwork (RSN))
- Inference: `0D; 2D (x, y)` input dimension is inferred because the paper uses both vectorized instance bags (e.g., "$x_{i,j} \in \mathbb{R}^{1 \times d}$" in Section 3.2) and image instances ("gray-scale digit images of size  $28 \times 28$ " in Section 4.1.2; "each image is of size 32 × 32" in Section 4.1.3; " $27\times27$  patches" in Section 4.1.4). `Capped` input dynamics is inferred from explicit variable but finite bag sizes (e.g., "different number of instances (3, 4 and 5)" in Fig. 3 caption context and "variable number of instances" in Section 3.5). `Dynamic` attention is inferred from runtime representation selection ("RSN aims to select one of the representation vectors" in Section 3.6). `0D` output dimension and `Fixed` output dynamics are inferred from single-label bag prediction ("The objective is to predict a bag target label" in Section 3.1).
