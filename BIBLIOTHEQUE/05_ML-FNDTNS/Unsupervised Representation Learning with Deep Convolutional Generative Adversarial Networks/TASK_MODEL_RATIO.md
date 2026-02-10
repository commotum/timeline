1. **Number of distinct tasks evaluated:** 3

   - Generative modeling: “We trained DCGANs on three datasets, Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset.” (Section 4, “DETAILS OF ADVERSARIAL TRAINING”)
   - Classification task 1: “#### 5.1 Classifying CIFAR-10 using GANs as a feature extractor” (Section 5.1)
   - Classification task 2: “#### 5.2 Classifying SVHN digits using GANs as a feature extractor” (Section 5.2)

2. **Number of trained model instances required to cover all tasks:** 3 models (minimum explicitly required)

   - One trained DCGAN model is required for generative modeling/feature extraction: “We trained DCGANs on three datasets...” (Section 4)
   - One CIFAR-10 task-specific classifier is separately trained: “a regularized linear L2-SVM classifier is trained on top of them.” (Section 5.1)
   - One SVHN task-specific classifier is separately trained: “used to train a regularized linear L2-SVM classifier on top of the same feature extraction pipeline used for CIFAR-10.” (Section 5.2)
   - Whether CIFAR-10 and SVHN use the same single trained DCGAN backbone is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
