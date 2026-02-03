1. Number of distinct tasks evaluated: 3 (unconditioned image generation, class-conditioned image generation, image super-resolution). Evidence: “Our unconditioned and class-conditioned image generation models both use 1D local attention,” (Section 5.1. Generative Image Modeling). “We trained the class-conditioned Image Transformer on CIFAR-10” (Section 5.2. Conditioning on Image Class). “Super-resolution is the process of recovering a high resolution image from a low resolution image while generating realistic and plausible details.” (Section 5.3. Image Super-Resolution).
2. Number of trained model instances required to cover all tasks: 3. Evidence: “Our unconditioned and class-conditioned image generation models both use 1D local attention,” (Section 5.1. Generative Image Modeling). “We trained the class-conditioned Image Transformer on CIFAR-10” (Section 5.2. Conditioning on Image Class). “We perform end-to-end training of the encoder-decoder model for Super resolution” (Section 5.3. Image Super-Resolution).
3. Task-Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
