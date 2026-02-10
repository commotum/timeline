1. **Number of distinct tasks evaluated:** 5.

"To disentangle digit shape from styles on MNIST, we choose to model the latent codes with one categorical code,  $c_1 \sim \operatorname{Cat}(K=10,p=0.1)$ , which can model discontinuous variation in data, and two continuous codes that can capture variations that are continuous in nature:  $c_2, c_3 \sim \operatorname{Unif}(-1,1)$ ." "Next we evaluate InfoGAN on two datasets of 3D images: faces [31] and chairs [32], on which DC-IGN was shown to learn highly interpretable graphics codes." "Next we evaluate InfoGAN on the Street View House Number (SVHN) dataset, which is significantly more challenging to learn an interpretable representation because it is noisy, containing images of variable-resolution and distracting digits, and it does not have multiple variations of the same object." "Finally we show in Figure 6 that InfoGAN is able to learn many visual concepts on another challenging dataset: CelebA [33], which includes 200,000 celebrity images with large pose variations and background clutter." (Section 7.2)

2. **Number of trained model instances required to cover all tasks:** 5 models.

"The details for each set of experiments are presented below." "For this task, we use 1 ten-dimensional categorical code, 2 continuous latent codes and 62 noise variables, resulting in a concatenated dimension of 74." "For this task, we use 4 ten-dimensional categorical code, 4 continuous latent codes and 124 noise variables, resulting in a concatenated dimension of 168." "For this task, we use 10 ten-dimensional categorical code and 128 noise variables, resulting in a concatenated dimension of 228." "For this task, we use 5 continuous latent codes and 128 noise variables, so the input to the generator has dimension 133." "For this task, we use 1 continuous latent code, 3 discrete latent codes (each with dimension 20), and 128 noise variables, so the input to the generator has dimension 189." (Appendix C, C.1-C.5)

3. **Task–Model Ratio**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
