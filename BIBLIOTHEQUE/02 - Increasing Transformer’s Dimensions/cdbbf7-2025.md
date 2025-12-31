# SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection (2025)
Source: cdbbf7-2025.pdf

## Core reasons
- Proposes a transformer-based detector for 3D object detection that directly uses sparse 3D voxel features.
- The central contribution is a 3D detection architecture that replaces BEV with sparse voxel features, adapting the model to a higher-dimensional domain.

## Evidence extracts
- "duce a novel sparse voxel-based transformer network for
3D object detection, dubbed as SparseVoxFormer. Instead
of performing BEV feature extraction, we directly lever-
age sparse voxel features as the input for a transformer-
baseddetector." (p. 1)
- "Asourworkpresentanewparadigmof3Dobjectdetection
architecture (Fig. 1), which directly utilizes sparse voxel
features instead of BEV features, for comprehensive un-
derstanding, we first present our basic architecture essen-
tial for handling sparse features and then describe more
sparsefeatures-specificarchitecture.Beforedelvingintothe
Ourapproach Distinctfromthepreviousapproachesthat
useBEVfeatures,wedirectlyfeedsparse3Dvoxelfeatures
into our 3D object detector (Fig. 2a)." (p. 4)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
