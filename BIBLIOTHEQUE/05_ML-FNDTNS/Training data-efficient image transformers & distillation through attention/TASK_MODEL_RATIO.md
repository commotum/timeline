1. **Number of distinct tasks evaluated:** 7

- "We show that our neural networks that contain no convolutional layer can achieve competitive results against the state of the art on ImageNet with no external data." (Section 1. Introduction)
- "Our models pre-learned on Imagenet are competitive when transferred to different downstream tasks such as fine-grained classification, on several popular public benchmarks: CIFAR-10, CIFAR-100, Oxford-102 flowers, Stanford Cars and iNaturalist-18/19." (Section 1. Introduction)
- "*Table 6.* We compare Transformers based models on different transfer learning task with ImageNet pre-training." and "| Model               | ImageNet | CIFAR-10 | CIFAR-100 | Flowers | Cars | iNat-18 | iNat-19 | im/sec |" (Section 5.4)

2. **Number of trained model instances required to cover all tasks:** 7

- "We evaluated this on transfer learning tasks by fine-tuning on the datasets in Table 8." (Section 5.4)
- Because the evaluated transfer tasks are dataset-specific benchmarks listed separately in Table 6 (ImageNet, CIFAR-10, CIFAR-100, Flowers, Cars, iNat-18, iNat-19), covering all of them requires separate fine-tuned model instances.

3. **Task–Model Ratio**

$$
\boxed{
\frac{7\ \text{tasks}}{7\ \text{models}} = 1
}
$$
