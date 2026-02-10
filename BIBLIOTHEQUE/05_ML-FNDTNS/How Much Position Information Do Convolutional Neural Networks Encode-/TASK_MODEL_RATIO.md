1. **Number of distinct tasks evaluated:** 3

   - "our goal is to predict a gradient-like position information mask" (Section 2, Problem Formulation).
   - "Saliency Detection: We further validate our findings in the position-dependent tasks (semantic segmentation and salient object detection (SOD))." (Section 4.2, Zero-Padding Driven Position Information).
   - "Semantic Segmentation: We also validate the impact of zero-padding on the semantic segmentation task." (Section 4.2, Zero-Padding Driven Position Information).

2. **Number of trained model instances required to cover all tasks:** 3 models

   - "we train the VGG and ResNet based networks on each type of the ground-truth" (Section 3.3, Existence of Position Information).
   - "we train two VGG models on the tasks of semantic segmentation and saliency detection from scratch, denoted as VGG-SS and VGG-SOD respectively. Then we finetune these three VGG models" (Section 4.2, Zero-Padding Driven Position Information).

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
