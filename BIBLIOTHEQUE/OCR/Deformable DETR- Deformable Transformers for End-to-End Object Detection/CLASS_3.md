# DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION (Not specified in the paper.)
Source: Deformable DETR- Deformable Transformers for End-to-End Object Detection.md

## Core reasons
- Identifies standard Transformer attention over image feature maps as slow and resolution-limited, motivating a new computation mechanism.
- Introduces deformable attention that changes how attention is computed by sampling a small set of key points, improving efficiency and convergence.

## Evidence extracts
- "However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps." (Abstract)
- "the deformable attention module only attends to a small set of key sampling points around a reference point, regardless of the spatial size of the feature maps, as shown in Fig. 2." (Section 4.1)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
