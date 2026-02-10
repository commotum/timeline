1. **Number of distinct tasks evaluated:** 3

- Classification: "### 4.1. ImageNet Classification" and "Table 6. Classification error on the **CIFAR-10** test set." (Sections 4.1 and 4.2)
- Object detection: "### 4.3. Object Detection on PASCAL and MS COCO" (Section 4.3)
- Localization: "# C. ImageNet Localization" and "The ImageNet Localization (LOC) task [36] requires to classify and localize the objects." (Section C)
- "COCO segmentation" is mentioned (Abstract/Introduction/Section 4.3), but evaluation details are: "Not specified in the paper."

2. **Number of trained model instances required to cover all tasks:** 3

- Detection requires a task-specific trained instance: "The models are initialized by the ImageNet classification models, and then fine-tuned on the object detection data." (Section A. Object Detection Baselines)
- Localization requires a task-specific trained instance: "We pre-train the networks for ImageNet classification and then fine-tune them for localization." (Section C. ImageNet Localization)
- Thus, to cover all distinct evaluated tasks (classification, detection, localization), separate trained instances are required for each task.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
