1. **Number of distinct tasks evaluated: 3**

"We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics." (Abstract)

"**Progressive lossy compression** We can probe further into the rate-distortion behavior of our model by introducing a progressive lossy code that mirrors the form of Eq. (5): see Algorithms 3 and 4, which assume access to a procedure, such as minimal random coding [19, 20], that can transmit a sample  $\mathbf{x} \sim q(\mathbf{x})$  using approximately  $D_{\mathrm{KL}}(q(\mathbf{x}) \parallel p(\mathbf{x}))$  bits on average for any distributions p and q, for which only p is available to the receiver beforehand." (Section 4.3, Progressive coding)

"We can interpolate source images  $\mathbf{x}_0, \mathbf{x}_0' \sim q(\mathbf{x}_0)$  in latent space using q as a stochastic encoder,  $\mathbf{x}_t, \mathbf{x}_t' \sim q(\mathbf{x}_t|\mathbf{x}_0)$ , then decoding the linearly interpolated latent  $\bar{\mathbf{x}}_t = (1-\lambda)\mathbf{x}_0 + \lambda\mathbf{x}_0'$  into image space by the reverse process,  $\bar{\mathbf{x}}_0 \sim p(\mathbf{x}_0|\bar{\mathbf{x}}_t)$ ." (Section 4.4, Interpolation)

2. **Number of trained model instances required to cover all tasks: 5**

"Our CIFAR10 model has 35.7 million parameters, and our LSUN and CelebA-HQ models have 114 million parameters." (Appendix B, Experimental details)

"We trained on CelebA-HQ for 0.5M steps, LSUN Bedroom for 2.4M steps, LSUN Cat for 1.8M steps, and LSUN Church for 1.2M steps." (Appendix B, Experimental details)

"Figure 5: Unconditional CIFAR10 test set rate-distortion vs. time." (Section 4.3, Progressive coding)

"Fig. 8 (right) shows interpolations and reconstructions of original CelebA-HQ  $256 \times 256$  images (t = 500)." (Section 4.4, Interpolation)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{5\ \text{models}} = 0.6
}
$$
