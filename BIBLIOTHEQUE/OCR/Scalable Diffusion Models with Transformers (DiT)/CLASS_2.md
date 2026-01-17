# Scalable Diffusion Models with Transformers (Not specified in the paper.)
Source: Scalable Diffusion Models with Transformers (DiT).md

## Core reasons
- Introduces Diffusion Transformers that replace the U-Net backbone in image diffusion with a transformer operating on latent image patches, making a transformer the primary backbone for a 2D visual domain.
- Uses ViT-style patch tokenization of spatial latents into sequences, which is a transformer adaptation to model higher-dimensional image data.

## Evidence extracts
- "We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches." (Abstract)
- "DiT is based on the Vision Transformer (ViT) architecture which operates on sequences of patches [10]." (Section 3.2)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
