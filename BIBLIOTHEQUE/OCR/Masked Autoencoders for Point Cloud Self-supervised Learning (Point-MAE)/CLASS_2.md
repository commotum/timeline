# Masked Autoencoders for Point Cloud Self-supervised Learning (Not specified in the paper)
Source: Masked Autoencoders for Point Cloud Self-supervised Learning (Point-MAE).md

## Core reasons
- Introduces a masked autoencoder for point clouds that uses a standard Transformer-based autoencoder backbone, extending transformer modeling to 3D point cloud data.
- Defines a 3D point cloud patching and embedding pipeline so transformer tokens can represent unordered 3D points for reconstruction.

## Evidence extracts
- "Then, a standard Transformer based autoencoder, with an asymmetric design and a shifting mask tokens operation, learns high-level latent features from unmasked point patches, aiming to reconstruct the masked point patches." (Abstract)
- "Unlike images in computer vision that can be naturally divided into regular patches, point cloud consists of unordered points in 3D space. Based on its property, we process the input point cloud through three stages: point patches generation, masking, and embedding." (Section 3.1 Point Cloud Masking and Embedding)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
