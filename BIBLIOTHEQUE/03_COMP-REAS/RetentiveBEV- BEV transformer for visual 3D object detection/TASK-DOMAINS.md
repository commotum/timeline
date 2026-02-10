# BEV transformer for visual 3D object detection applied with retentive mechanism (2025)
Source: RetentiveBEV- BEV transformer for visual 3D object detection.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | Multi-camera surround-view images; historical BEV features | 2D (x, y); 3D (x, y, t) | Fixed | Dynamic | Constructed | 3D object bounding boxes | 3D (x, y, z) | Capped (inferred) |
| Map segmentation | Multi-camera surround-view images; historical BEV features | 2D (x, y); 3D (x, y, t) | Fixed | Dynamic | Constructed | BEV map segmentation labels | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper covers two autonomous-driving perception tasks: 3D object detection and map segmentation, both built on BEV features from multi-camera visual input with temporal fusion. Inputs are image-centric and temporally enriched, supporting 2D spatial structure and spatiotemporal processing. The attention mechanism is Dynamic, and the model relies on Constructed state through learned BEV representations and historical BEV memory. Output domains span 3D detection geometry and BEV map segmentation, with some output-structure details inferred from fixed BEV grid design.

## Evidence
### Task: 3D object detection
- "The integration of the retentive mechanism notably boosts the precision and recall in 3D object detection while also expediting the inference process." (Section Abstract)
- "For image inputs from multiple cameras, where the nuScenes data set provides surround-view images from  $N_{ref} = 6$  cameras, the network starts by extracting multi-dimensional features from each camera through a backbone network." (Section Architecture)
- "These features are then integrated with temporal information from historical BEV frames via a Temporal Self-Attention (TSA) module and fed into an RSCA module for spatial feature aggregation." (Section Architecture)
- Inference: Out Dynamics is labeled Capped (inferred) because the model uses a bounded BEV query interface: "On the nuScenes data set, BEV queries were defaulted to a resolution of  $200 \times 200$ ." (Section Environment settings and baseline)

### Task: Map segmentation
- "Three-dimensional (3D) vision perception tasks utilizing multiple cameras are pivotal for autonomous driving systems, encompassing both 3D object detection and map segmentation." (Section Abstract)
- "This comprehensive process of \"temporal information aggregation, spatial feature aggregation, forward propagation\" is executed six times across the network, culminating in high-dimensional BEV features  $B_t$  for subsequent object detection or segmentation tasks." (Section Architecture)
- "To effectively assess the RetentiveBEV neck network's performance, we included VPN, Lift-Splat, and BEVFormer as benchmarks, applying the same head network for tasks of BEV detection and segmentation." (Section Environment settings and baseline)
- Inference: Out Dimension is labeled 2D (x, y) (inferred) and Out Dynamics is labeled Fixed (inferred) because map segmentation is described in BEV-grid form and the grid is explicitly fixed in configuration: "On the nuScenes data set, BEV queries were defaulted to a resolution of  $200 \times 200$ ." and "Each BEV grid's resolution (s) matches a real-world square region with sides of 0.512 m." (Section Environment settings and baseline)
