1. **Number of distinct tasks evaluated:** 1 task (image classification)

> "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes." (Abstract)
>
> "Our results on ILSVRC-2010 are summarized in Table 1. Our network achieves top-1 and top-5 test set error rates of **37.5%** and **17.0%**<sup>5</sup>." (Section 6 Results)

2. **Number of trained model instances required to cover all tasks:** 1 model

> "The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels." (Section 3.5 Overall Architecture)
>
> "The CNN described in this paper achieves a top-5 error rate of 18.2%." (Section 6 Results)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
