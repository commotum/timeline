# DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries (Not specified in the paper.)
Source: DETR3D- 3D Object Detection from Multi-view Images via 3D-to-2D Queries.md

## Core reasons
- Proposes a Transformer-style set prediction architecture for 3D object detection, extending DETR-like attention and object queries beyond 2D image space.
- The core contribution adapts attention-based object queries to operate in 3D by projecting 3D reference points into multi-view 2D features.

## Evidence extracts
- "Our architecture extracts 2D features from multiple camera images and then uses a sparse set of 3D object queries to index into these 2D features, linking 3D positions to multi-view images using camera transformation matrices." (Abstract)
- "Each object query encodes a 3D location, which is projected to the camera planes and used to collect image features via bilinear interpolation. Similarly to DETR [10], we then use multi-head attention [9] to refine the object queries by incorporating object interactions." (Section 3 Multi-view 3D Object Detection, 3.1 Overview)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
