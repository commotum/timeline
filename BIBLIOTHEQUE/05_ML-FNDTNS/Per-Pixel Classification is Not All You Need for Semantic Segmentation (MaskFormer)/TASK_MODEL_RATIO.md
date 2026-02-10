1. **Number of distinct tasks evaluated:** 2

> "We demonstrate that MaskFormer seamlessly unifies semantic- and instance-level segmentation tasks by showing state-of-the-art results on both semantic segmentation and panoptic segmentation datasets." (Section 4 Experiments)

2. **Number of trained model instances required to cover all tasks:** 2

> "Panoptic segmentation. We follow exactly the same architecture, loss, and training procedure as we use for semantic segmentation. The only difference is supervision: *i.e.*, category region masks in semantic segmentation vs. object instance masks in panoptic segmentation." (Section 4.2 Training settings)

Single jointly trained model instance that simultaneously covers all evaluated tasks without task-specific training: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
