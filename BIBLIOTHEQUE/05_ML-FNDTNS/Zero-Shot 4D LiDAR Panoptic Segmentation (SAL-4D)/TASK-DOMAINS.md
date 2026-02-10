# Zero-Shot 4D Lidar Panoptic Segmentation (Not specified in the paper.)
Source: Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Joint segmentation, tracking, and zero-shot recognition (Zero-Shot 4D Lidar Panoptic Segmentation) | 4D Lidar point cloud sequences; optional semantic vocabulary text prompts | 4D (x, y, z, t); 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Per-point spatio-temporal instance assignments and optional semantic class assignments | 4D (x, y, z, t) | Open (inferred) |

## Summary
The paper covers a single integrated task: zero-shot 4D Lidar panoptic segmentation that jointly performs segmentation, tracking, and recognition. The core modality is spatiotemporal Lidar point cloud sequences (4D), with optional text prompts for zero-shot class specification (mapped to 1D text sequence input, inferred). The input/output sequence behavior is treated as Open (inferred) because the method describes near-online cross-time association for arbitrary-length sequences rather than one isolated scan. Attention is classified as Static (inferred) and state as Constructed (inferred), based on fixed windowed processing plus explicit construction of instance identities and semantic track features.

## Evidence
### Task: Joint segmentation, tracking, and zero-shot recognition (Zero-Shot 4D Lidar Panoptic Segmentation)
- "We tackle segmentation, tracking, and zero-shot recognition of any object in Lidar sequences." (Section 1. Introduction)
- "**Zero-shot 4D Lidar panoptic segmentation.** We address 4D-LPS in a zero-shot setting, intending to localize and recognize *any* objects in 4D Lidar point cloud sequences." (Section 3.1. Problem Statement)
- "Similarly, we assign *each* points  $p \in \mathcal{P}$  an instance identity id  $\in \mathbb{N}$ ; however, we do not assume predefined semantic class vocabulary and (accordingly) labeled training set at train time. Instead, we assume a semantic vocabulary  $\mathcal{C}_{test}$  is *optionally* specified at test-time as a list of free-form descriptions of semantic classes. When specified, we assign points also to semantic classes  $c \in \mathcal{C}_{test}$ ." (Section 3.1. Problem Statement)
- "As our model directly processes superimposed point clouds within windows of size K, we perform *near-online* inference [15] by associating Lidar masklets across time based on 3D-IoU overlap via bi-partite matching (as described in Sec. 3.2.2). For zero-shot prompting, we follow [62] and first encode prompts specified in the semantic class vocabulary using a CLIP language encoder." (Section 3.3. SAL-4D Model, Inference)
- Inference: `1D (t)` input is inferred from the paper’s text-prompted semantic vocabulary interface (free-form descriptions/prompts). `Open` input/output dynamics are inferred from the near-online cross-window association over sequences (not limited to one standalone frame). `Static` attention is inferred because the model consumes predefined windowed inputs and prompt sets without an explicit runtime retrieval/selection policy. `Constructed` state is inferred from explicit construction/maintenance of object-centric representations and identities (e.g., per-query masks/objectness/CLIP tokens and cross-window instance-ID updates).
