1. **Number of distinct tasks evaluated:** 3

"We evaluate our method on the ImageNet 2012 classification dataset [36] that consists of 1000 classes." (Section 4.1. ImageNet Classification)

"Table 6. Classification error on the **CIFAR-10** test set." (Section 4.2. CIFAR-10 and Analysis)

"Table 7 and 8 show the object detection baseline results on PASCAL VOC 2007 and 2012 [5] and COCO [26]." (Section 4.3. Object Detection on PASCAL and MS COCO)

"The ImageNet Localization (LOC) task [36] requires to classify and localize the objects." (Section C. ImageNet Localization)

2. **Number of trained model instances required to cover all tasks:** 3 models

"The models are initialized by the ImageNet classification models, and then fine-tuned on the object detection data." (Section A. Object Detection Baselines)

"We pre-train the networks for ImageNet classification and then fine-tune them for localization." (Section C. ImageNet Localization)

"The final classification layer is replaced by two sibling layers (classification and box regression [7])." (Section A. Object Detection Baselines)

"This RPN ends with two sibling  $1\times1$  convolutional layers for binary classification (*cls*) and box regression (*reg*), as in [32]." (Section C. ImageNet Localization)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
