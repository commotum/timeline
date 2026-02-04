1. Number of distinct tasks evaluated: 2.
> "Waymo Open Dataset (WOD). This dataset contains 230,000 annotated samples split into 160,000 for training, 40,000 for validation, and 30,000 for testing. It uses two metrics for 3D object detection: mean average precision (mAP) and mAP weighted by heading accuracy (mAPH), further categorized into Level 1 (L1) for objects detected by more than five LiDAR points and Level 2 (L2) for those detected with at least one point." (Section 4.1 Datasets and Evaluation Metrics)
>
> "**NuScenes.** This dataset comprises 40,000 annotated samples, with 28,000 for training, 6,000 for validation, and 6,000 for testing. On this dataset, the model performance is measured by mean average precision (mAP) across multiple distance thresholds (0.5, 1, 2, and 4 meters) and the nuScenes detection score (NDS), which combines mAP with a weighted sum of five additional metrics assessing true positive predictions in translation, scale, orientation, velocity, and attribute accuracy." (Section 4.1 Datasets and Evaluation Metrics)

2. Number of trained model instances required to cover all tasks: 2.
> "To construct ScatterFormer, we set the voxel size to (0.32m, 0.32m, 0.1875m) for the Waymo dataset and (0.3m, 0.3m, 8m) for the NuScenes dataset." (Section 4.2 Implementation Details)
>
> "ScatterFormer is trained for 24 epochs with a learning rate of 0.006 on Waymo Dataset and 20 epochs with a learning rate of 0.004 on NuScenes Dataset." (Section 4.2 Implementation Details)

3. Task–Model Ratio = (1) / (2).

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
