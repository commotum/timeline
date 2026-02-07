# Point Transformer V3: Simpler, Faster, Stronger (Not specified in the paper.)
Source: Point Transformer V3- Simpler, Faster, Stronger.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic segmentation (indoor) | indoor 3D point clouds | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | per-point semantic labels (inferred) | 3D (x, y, z) (inferred) | Not specified in the paper. |
| semantic segmentation (outdoor) | outdoor 3D point clouds | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | per-point semantic labels (inferred) | 3D (x, y, z) (inferred) | Not specified in the paper. |
| instance segmentation (indoor) | indoor 3D point clouds | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | per-point instance labels (inferred) | 3D (x, y, z) (inferred) | Not specified in the paper. |
| object detection (outdoor) | outdoor 3D point clouds | 3D (x, y, z) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | 3D object bounding boxes (inferred) | 3D (x, y, z) (inferred) | Not specified in the paper. |

## Summary
PTv3 is evaluated on 3D point cloud perception tasks spanning indoor and outdoor semantic segmentation, indoor instance segmentation, and outdoor object detection. The paper defines point positions in 3D space, so inputs are 3D (x, y, z) point clouds and outputs are 3D-indexed labels or detections (inferred from task names). Attention is described as patch attention over non-overlapping patches and the model uses a U-Net style encoder-decoder, so attention is treated as static and state as constructed (inferred). The paper does not specify fixed or capped input/output size constraints, and it additionally reports data-efficiency results within the indoor ScanNet setting.

## Evidence
### Task: semantic segmentation (indoor)
- "We report the performance using the mean results from the ScanNet semantic segmentation validation" (Section 5.1. Main Properties)
- "Table 5. Indoor semantic segmentation." (Table 5)
- Inference: Output treated as per-point semantic labels and 3D-indexed; attention as Static; state as Constructed based on "$p_i \\in \\mathbb{R}^3$", "groups points into non-overlapping patches", and "The architecture of PTv3 remains consistent with the U-Net [66] framework." (Sections 4.1, 4.2, 4.3)

### Task: semantic segmentation (outdoor)
- "**Outdoor semantic segmentation.** In Tab. 7, we detail the validation and test results of PTv3" (Section 5.2. Results Comparision)
- "Table 7. Outdoor semantic segmentation." (Table 7)
- Inference: Output treated as per-point semantic labels and 3D-indexed; attention as Static; state as Constructed based on "$p_i \\in \\mathbb{R}^3$", "groups points into non-overlapping patches", and "The architecture of PTv3 remains consistent with the U-Net [66] framework." (Sections 4.1, 4.2, 4.3)

### Task: instance segmentation (indoor)
- "Indoor instance segmentation. In Tab. 8, we present PTv3's validation results on the ScanNet v2 [17] and ScanNet200 [67] instance segmentation benchmarks." (Section 5.2. Results Comparision)
- "Table 8. Indoor instance segmentation." (Table 8)
- Inference: Output treated as per-point instance labels and 3D-indexed; attention as Static; state as Constructed based on "$p_i \\in \\mathbb{R}^3$", "groups points into non-overlapping patches", and "The architecture of PTv3 remains consistent with the U-Net [66] framework." (Sections 4.1, 4.2, 4.3)

### Task: object detection (outdoor)
- "Outdoor object detection. In Tab. 10, we benchmark PTv3 against leading single-stage 3D detectors on the Waymo Object Detection benchmark." (Section 5.2. Results Comparision)
- "Table 10. **Waymo object detection.**" (Table 10)
- Inference: Output treated as 3D object bounding boxes and 3D-indexed; attention as Static; state as Constructed based on "$p_i \\in \\mathbb{R}^3$", "groups points into non-overlapping patches", and "The architecture of PTv3 remains consistent with the U-Net [66] framework." (Sections 4.1, 4.2, 4.3)
