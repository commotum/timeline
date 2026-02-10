1. **Number of distinct tasks evaluated:** 4

> "We empirically test our architecture on density modeling tasks including natural images, text, and raw audio." (Section 7. Experiments)
>
> "#### 7.1. CIFAR-10" (Section 7.1)
>
> "#### **7.2.** Text" (Section 7.2)
>
> "#### 7.3. ImageNet 64x64" (Section 7.3)
>
> "#### 7.4. Classical music from raw audio" (Section 7.4)

2. **Number of trained model instances required to cover all tasks:** 4

> "We train strided Sparse Transformers on CIFAR-10 images represented as sequences of 3072 bytes." (Section 7.1)
>
> "In order to assess Sparse Transformers on datasets without a strong two-dimensional structure, we trained models on the EnWik8 dataset..." (Section 7.2)
>
> "In order to test the ability of the model to learn long range dependencies and scale to a large dataset, we train on the version of downsampled ImageNet released by (Oord et al., 2016) and evaluate on the validation set." (Section 7.3)
>
> "To test the extent to which Sparse Transformers are able to scale to very long contexts, we trained models on the classical music dataset released by (Dieleman et al., 2018)." (Section 7.4)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
