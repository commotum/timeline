# Transformer-based Point Cloud Generation Network (2023)
Source: 98b726-2023.pdf

## Core reasons
- The paper introduces a transformer-based point cloud generator that explicitly targets 3D data, fitting the goal of increasing Transformer applicability beyond 1D language tasks.
- Transformer-based interpolation and refinement modules stitch latent vectors into structured 3D coordinates, showing the core contribution is adapting the transformer's architecture to handle spatial geometry rather than retooling positional embeddings.

## Evidence extracts
- "Point cloud generation is an important research topic in 3D computer vision, which can provide high-quality datasets for various downstream tasks. However, efficiently capturing the geometry of point clouds remains a challenging problem due to their irregularities. In this paper, we propose a novel transformer-based 3D point cloud generation network to generate realistic point clouds. Specifically, we first develop a transformer-based interpolation module that utilizes k-nearest neighbors at different scales to learn global and local information about point clouds in the feature space. Based on geometric information, we interpolate new point features to upsample the point cloud features. Then, the upsampled features are used to generate a coarse point cloud with spatial coordinate information. We construct a transformer-based refinement module to enhance the upsampled features in feature space with geometric information in coordinate space. Finally, we use a multi-layer perceptron on the upsampled features to generate the final point cloud." (Abstract)
- "Our model aims to generate high-quality point clouds from a latent vector input. Using a fully-connected layer, we first map the latent vector to an initial point cloud feature map. To upsample the point cloud feature, we use the transformer-based interpolation (TIM) module to interpolate in feature space. The upsampled feature is then used to generate a coarse point cloud with an underlying structure using MLP. Additionally, we refine the upsampled feature based on the geometric structure of the generated rough point cloud in coordinate space. Finally, we use a max-pooling aggregation [on] the refined feature, and an MLP to generate the final point cloud coordinates." (Section 3.1)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
