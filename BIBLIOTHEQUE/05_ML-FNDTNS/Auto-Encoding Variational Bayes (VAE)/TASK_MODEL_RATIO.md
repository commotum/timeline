1. **Number of distinct tasks evaluated:** 3

> "We are interested in, and propose a solution to, three related problems in the above scenario:" (Section 2.1, *Problem scenario*)
>
> "- 1. Efficient approximate ML or MAP estimation for the parameters  $\theta$ ."
>
> "- 2. Efficient approximate posterior inference of the latent variable z given an observed value x for a choice of parameters  $\theta$ ."
>
> "- 3. Efficient approximate marginal inference of the variable x."

2. **Number of trained model instances required to cover all tasks:** 1

> "The variational parameters  $\phi$  are learned jointly with the generative model parameters  $\theta$ ." (Figure 1 caption, Section 2, *Method*)
>
> "In this section we'll give an example where we use a neural network for the probabilistic encoder  $q_{\phi}(\mathbf{z}|\mathbf{x})$  (the approximation to the posterior of the generative model  $p_{\theta}(\mathbf{x}, \mathbf{z})$ ) and where the parameters  $\phi$  and  $\theta$  are optimized jointly with the AEVB algorithm." (Section 3, *Example: Variational Auto-Encoder*)
>
> "The learned approximate posterior inference model can also be used for a host of tasks such as recognition, denoising, representation and visualization purposes." (Section 1, *Introduction*)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$
