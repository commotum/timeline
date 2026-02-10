1. **Number of distinct tasks evaluated:** 3

"We investigated the use of black-box ODE solvers as a model component, developing new models for time-series modeling, supervised learning, and density estimation." (Section 8, Conclusion)

"In this section, we experimentally investigate the training of neural ODEs for supervised learning." (Section 3)

"This lets us train the flow on a density estimation task by performing maximum likelihood estimation, which maximizes  $\mathbb{E}_{p(\mathbf{x})}[\log q(\mathbf{x})]$  where  $q(\cdot)$  is computed using the appropriate change of variables theorem, then afterwards reverse the CNF to generate random samples from  $q(\mathbf{x})$ ." (Section 4.1, Experiments with Continuous Normalizing Flows)

"We investigate the ability of the latent ODE model to fit and extrapolate time series." (Section 5.1, Time-series Latent ODE Experiments)

2. **Number of trained model instances required to cover all tasks:** 3

"Model Architectures We experiment with a small residual network which downsamples the input twice then applies 6 standard residual blocks He et al. (2016b), which are replaced by an ODESolve module in the ODE-Net variant." (Section 3)

"We call these models continuous normalizing flows (CNF)." (Section 4)

"We present a continuous-time, generative approach to modeling time series." (Section: A generative latent function time-series model)

A single jointly trained model instance that performs all three task types is Not specified in the paper.

3. **Task–Model Ratio**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
