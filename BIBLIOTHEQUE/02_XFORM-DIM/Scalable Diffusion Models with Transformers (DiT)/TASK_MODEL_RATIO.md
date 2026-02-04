1. Number of distinct tasks evaluated: 2 (class-conditional ImageNet generation at 256×256 and 512×512). Evidence: “We train class-conditional latent DiT models at  $256 \times 256$  and  $512 \times 512$  image resolution” (Section 4. Experimental Setup).
2. Number of trained model instances required to cover all tasks: 2. Evidence: “256×256 ImageNet. Following our scaling analysis, we continue training our highest Gflop model, DiT-XL/2, for 7M steps.” (Section 5.1. State-of-the-Art Diffusion Models) and “512×512 ImageNet. We train a new DiT-XL/2 model on ImageNet at  $512 \times 512$  resolution for 3M iterations” (Section 5.1. State-of-the-Art Diffusion Models).
3. Task–Model Ratio:

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
