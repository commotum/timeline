1. Number of distinct tasks evaluated: 3. "With this intention, we propose Mask4Former for the challenging task of 4D panoptic segmentation of LiDAR point clouds." (Abstract) "Specifically, we use Mask4Former for both 3D panoptic segmentation and 4D semantic segmentation tasks." (Supplementary Material)
2. Number of trained model instances required to cover all tasks: 3. "Transitioning from 4D to 3D panoptic segmentation for Mask4Former is straightforward by adjusting the number of superimposed LiDAR scans to 1." (Supplementary Material) "Transitioning from 4D panoptic segmentation to 4D semantic segmentation requires two minor modifications." (Supplementary Material) "Firstly, instead of generating a target mask for each instance, a single target mask per class is generated." (Supplementary Material) "Secondly, bounding box parameter regression is omitted since a single target mask may encompass multiple instances of the same class." (Supplementary Material)
3. Task-Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
