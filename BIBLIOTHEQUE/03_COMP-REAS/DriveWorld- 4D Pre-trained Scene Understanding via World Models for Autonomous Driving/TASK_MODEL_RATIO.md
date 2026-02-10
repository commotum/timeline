1. **Number of distinct tasks evaluated:** 6

   "Our 4D pre-training algorithm exhibited substantial improvements in vision-centric autonomous driving tasks, including 3D object detection, multi-object tracking, online mapping, motion forecasting, occupancy prediction, and planning." (Section 1. Introduction)

2. **Number of trained model instances required to cover all tasks:** 2

   "For the 3D detection task, we employed the BEVFormer [44] framework, fine-tuning its parameters without freezing the encoder, and conducted training for 24 epochs." (Section 4.1. Experimental Setup, Fine-tuning)

   "Regarding other autonomous driving tasks, we utilized the UniAD [31] framework and loaded our fine-tuned BEVFormer weights to UniAD, adhering to a standard 20-epoch training protocol for all tasks." (Section 4.1. Experimental Setup, Fine-tuning)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{6\ \text{tasks}}{2\ \text{models}} = 3
}
$$
