# Mask4D: End-to-End Mask-Based 4D Panoptic Segmentation for LiDAR Sequences (2023)
Source: 942ef3-2023.pdf

## Core reasons
- The paper adapts a transformer-based mask segmentation model to 4D LiDAR sequences by reusing queries over scans so that each query carries the same instance ID across time, turning the architecture into a temporal model without any post-processing.
- Detection queries decode new/different scans while tracking queries carry identities forward, providing consistent instance IDs and allowing the network to output per-point semantics and identities directly for each 4D scan pair.

## Evidence extracts
- "We extend a mask-based 3D panoptic segmentation model to 4D by reusing queries that decoded instances in previous scans. This way, each query decodes the same instance over time, carries its ID and the tracking is performed implicitly." (p. 1)
- "In our proposed approach, we use two groups of queries as input: detection queries Q_det and tracking queries Q_tr, and we input them simultaneously into the network at each step together with the LiDAR scan. New instances and the stuff classes are decoded by the fixed N detection queries Q_det while the already tracked instances are decoded by their corresponding tracking queries Q_tr and thus keep a consistent instance ID. This way, we do not perform any association or post-processing, and our approach directly outputs for each point a semantic class and instance IDs which are consistent over time." (Sec. III-B)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
