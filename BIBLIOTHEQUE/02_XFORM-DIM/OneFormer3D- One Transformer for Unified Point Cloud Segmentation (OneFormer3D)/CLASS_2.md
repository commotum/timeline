# OneFormer3D: One Transformer for Unified Point Cloud Segmentation (Not specified in the paper)
Source: aa7b28-2024.pdf

## Core reasons
- Proposes a unified 3D point cloud segmentation framework covering semantic, instance, and panoptic tasks, applying transformer modeling to 3D data.
- Uses a transformer-based decoder with unified instance and semantic queries to generate segmentation masks, indicating an architectural adaptation rather than positional encoding changes or dataset creation.

## Evidence extracts
- "These kernels are trained with a transformer-
based decoder with unified instance and semantic queries
passed as an input." (p. 1)
- "Taking a 3D point
cloudasinput,ourtrainedmodelsolves3Dinstance,3Dsemantic,and3Dpanopticsegmentationtasks." (p. 3)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
