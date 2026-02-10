1. **Number of distinct tasks evaluated:** 3

   - "We demonstrate our approach on high-dimensional density estimation, image generation, and variational inference, achieving the state-of-the-art among exact likelihood methods with efficient sampling." (Abstract)
   - "We demonstrate the power of FFJORD on a variety of density estimation tasks as well as approximate inference within variational autoencoders (Kingma & Welling, 2014)." (Section 4 EXPERIMENTS)

2. **Number of trained model instances required to cover all tasks:** 2 models

   - "We perform density estimation on five tabular datasets preprocessed as in Papamakarios et al. (2017) and two image datasets; MNIST and CIFAR10." (Section 4.2 Density Estimation on Real Data)
   - "We compare FFJORD to other normalizing flows for use in variational inference. We train a VAE (Kingma & Welling, 2014) on four datasets using a FFJORD flow..." (Section 4.3 VARIATIONAL AUTOENCODER)
   - "In VAEs it is common for the encoder network to also output the parameters of the flow as a function of the input x... Instead, the encoder network outputs a low-rank update to a global weight matrix and an input-dependent bias vector." (Section 4.3 VARIATIONAL AUTOENCODER)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{2\ \text{models}} = 1.5
}
$$
