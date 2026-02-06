# BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers (Not specified in the paper.)
Source: BEVFormer- Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D object detection | multi-camera images | 3D (x, y, t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | 3D bounding boxes and velocity | 3D (x, y, z) | Not specified in the paper. |
| map segmentation | multi-camera images | 3D (x, y, t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | BEV semantic map (car/vehicles/road/lane) | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper presents BEVFormer as a multi-camera, spatiotemporal perception framework that supports two tasks: 3D object detection and BEV map segmentation. Inputs are multi-camera image sequences sampled across timestamps, and outputs are 3D bounding boxes/velocity or BEV semantic maps. The model uses spatial cross-attention over regions of interest and recurrently fuses history BEV features, implying dynamic attention and constructed state. For segmentation, outputs are BEV-grid maps derived from 2D BEV feature maps, while input sequence length is capped by the fixed four-sample training setup.

## Evidence
### Task: 3D object detection
- "3D visual perception tasks, including 3D detection and map segmentation based on multi-camera images, are essential for autonomous driving systems." (Abstract)
- "Each sample consists of RGB images from 6 cameras and has 360° horizontal FOV." (4.1 Datasets)
- "For 3D object detection, we design an end-to-end 3D detection head based on the 2D detector Deformable DETR [56]." (3.5 Applications of BEV Features)
- "The modifications include using single-scale BEV features  $B_t$  as the input of the decoder, predicting 3D bounding boxes and velocity rather than 2D bounding boxes, and only using  $L_1$  loss to supervise 3D bounding box regression." (3.5 Applications of BEV Features)
- "We denote the timestamps of these four samples as t-3, t-2, t-1 and t." (3.6 Implementation Details)
- "At the same time, we preserved the BEV features  $B_{t-1}$  at the prior timestamp t-1." (3.1 Overall Architecture)
- "To aggregate spatial information, we design spatial cross-attention that each BEV query extracts the spatial features from the regions of interest across camera views." (Abstract)
- Inference: In Dimension is 3D (x, y, t) (inferred) and In Dynamics is Capped (inferred) because the model explicitly uses multiple timestamps (t-3 to t) per sample; Attention Dynamic is Dynamic (inferred) because BEV queries extract regions of interest across camera views; State Dynamic is Constructed (inferred) because history BEV features are preserved and fused. (3.6 Implementation Details; 3.1 Overall Architecture; Abstract)

### Task: map segmentation
- "3D visual perception tasks, including 3D detection and map segmentation based on multi-camera images, are essential for autonomous driving systems." (Abstract)
- "Each sample consists of RGB images from 6 cameras and has 360° horizontal FOV." (4.1 Datasets)
- "**For map segmentation**, we design a map segmentation head based on a 2D segmentation method Panoptic SegFormer [22]." (3.5 Applications of BEV Features)
- "Since the map segmentation based on the BEV is basically the same as the common semantic segmentation, we utilize the mask decoder of [22] and class-fixed queries to target each semantic category, including the car, vehicles, road (drivable area), and lane." (3.5 Applications of BEV Features)
- "Since the BEV features  $B_t \in \mathbb{R}^{H \times W \times C}$  is a versatile 2D feature map that can be used for various autonomous driving perception tasks, the 3D object detection and map segmentation task heads can be developed based on 2D perception methods [56, 22] with minor modifications." (3.5 Applications of BEV Features)
- "We denote the timestamps of these four samples as t-3, t-2, t-1 and t." (3.6 Implementation Details)
- "At the same time, we preserved the BEV features  $B_{t-1}$  at the prior timestamp t-1." (3.1 Overall Architecture)
- "To aggregate spatial information, we design spatial cross-attention that each BEV query extracts the spatial features from the regions of interest across camera views." (Abstract)
- Inference: In Dimension is 3D (x, y, t) (inferred) and In Dynamics is Capped (inferred) because the model explicitly uses multiple timestamps (t-3 to t) per sample; Attention Dynamic is Dynamic (inferred) because BEV queries extract regions of interest across camera views; State Dynamic is Constructed (inferred) because history BEV features are preserved and fused; Out Dimension is 2D (x, y) (inferred) and Out Dynamics is Fixed (inferred) because map segmentation is based on BEV features described as a 2D feature map with shape  $H \times W$. (3.5 Applications of BEV Features; 3.6 Implementation Details; 3.1 Overall Architecture; Abstract)
