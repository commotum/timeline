# CamPoint: Boosting Point Cloud Segmentation with Virtual Camera (Not specified in the paper.)
Source: CamPoint- Boosting Point Cloud Segmentation with Virtual Camera.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic segmentation | 3D point cloud | 3D (x,y,z) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | per-point semantic labels | 3D (x,y,z) | Not specified in the paper. |
| object part segmentation | 3D point cloud | 3D (x,y,z) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | per-point part labels | 3D (x,y,z) | Not specified in the paper. |
| object classification | 3D point cloud | 3D (x,y,z) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | object class label (inferred) | 0D (inferred) | Not specified in the paper. |

## Summary
The paper covers point cloud semantic segmentation, object part segmentation, and 3D object classification, all operating on 3D point clouds with (x,y,z) coordinates. Segmentation outputs are per-point labels in the same 3D address space, while classification outputs a single class label (0D). Input/output size dynamics are not explicitly stated; the method’s data-dependent neighbor selection and constructed camera-based features suggest Dynamic attention and Constructed state (inferred).

## Evidence
### Task: semantic segmentation
- "In this section, we conduct experiments on point cloud semantic segmentation and point cloud object part segmentation tasks." (Section 4 Experiment)
- "our goal is to segment points based on their highest predicted classification scores." (Section 3 Method)
- "where  $s_i\in\mathbb{R}^3$  denotes point coordinates (x,y,z)" (Section 3 Method)
- Inference: Attention Dynamic = Dynamic (inferred) because the method "search the N most relevant neighbors" using KNN based on input distances, so the considered neighbors vary with the input. (Section 3.2)
- Inference: State Dynamic = Constructed (inferred) because it is "constructing the essential camera visibility feature for each point" to build internal representations. (Section 3 Method)

### Task: object part segmentation
- "In this section, we conduct experiments on point cloud semantic segmentation and point cloud object part segmentation tasks." (Section 4 Experiment)
- "Each 3D model is annotated into multiple parts (e.g., chair seat, backrest, armrest), with semantic labels for each part." (Section 4.1.2 Object Part Segmentation)
- "where  $s_i\in\mathbb{R}^3$  denotes point coordinates (x,y,z)" (Section 3 Method)
- Inference: Attention Dynamic = Dynamic (inferred) because the method "search the N most relevant neighbors" using KNN based on input distances, so the considered neighbors vary with the input. (Section 3.2)
- Inference: State Dynamic = Constructed (inferred) because it is "constructing the essential camera visibility feature for each point" to build internal representations. (Section 3 Method)

### Task: object classification
- "we also adapt CamPoint to object classification task" (Section 4.1.4 Object Classification)
- "ScanObjectNN [7] is a benchmark dataset for 3D object recognition and classification, consisting of 115 categories" (Section 4.1.4 Object Classification)
- "where  $s_i\in\mathbb{R}^3$  denotes point coordinates (x,y,z)" (Section 3 Method)
- Inference: Output = object class label (inferred) and Out Dimension = 0D (inferred) because the paper frames it as an "object classification task" with datasets defined by discrete "categories." (Section 4.1.4 Object Classification)
- Inference: Attention Dynamic = Dynamic (inferred) because the method "search the N most relevant neighbors" using KNN based on input distances, so the considered neighbors vary with the input. (Section 3.2)
- Inference: State Dynamic = Constructed (inferred) because it is "constructing the essential camera visibility feature for each point" to build internal representations. (Section 3 Method)
