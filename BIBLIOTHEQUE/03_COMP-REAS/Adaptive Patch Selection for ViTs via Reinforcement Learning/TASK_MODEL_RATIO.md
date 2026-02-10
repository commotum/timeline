1. **Number of distinct tasks evaluated:** 1

   - "We tested AgentViT on CIFAR10, FashionMNIST, and Imagenette<sup>+</sup> (which is a subset of ImageNet) in the image classification task and obtained promising performance when compared to baseline ViTs and other related approaches available in the literature." (Abstract)
   - "Finally, we emphasize that the overall framework is designed to support the integration of additional terms into the reward function. Thus, if another task beyond classification needs to be addressed, requiring a more sophisticated solution, a user can directly modify the reward function by adding the necessary terms while leaving the rest of the framework unchanged. This flexibility allows users to adapt AgentViT to tasks beyond the scope of this paper while preserving its core structure." (Section 3.4 Reward)
   - "For this purpose, we adopted the classical Markov Decision Process mechanism to represent an environment for the image classification task, which required us to redefine the state, action and reward necessary to train our agent." (Section 6 Conclusion)

2. **Number of trained model instances required to cover all tasks:** 1 model

   - "To thoroughly evaluate our approach, we used both the classical ViT [4] and SimpleViT [84]." (Section 4.1 Experimental setup)
   - "We tested AgentViT using ViT and SimpleViT as Vision Transformers, a Double Deep Q-Network as the internal agent, and applying it to CIFAR10, FashionMNIST, and Imagenette<sup>+</sup>." (Section 6 Conclusion)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{tasks}}{1\ \text{models}} = 1
}
$$
