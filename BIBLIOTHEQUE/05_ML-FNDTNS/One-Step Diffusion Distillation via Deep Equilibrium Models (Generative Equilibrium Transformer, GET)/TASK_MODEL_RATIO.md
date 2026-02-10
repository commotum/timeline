1. Number of distinct tasks evaluated: 2

> "We evaluate the effectiveness of our proposed Generative Equilibrium Transformer (GET) in offline distillation of diffusion models through a series of experiments on single-step class-conditional and unconditional image generation." (Section 4, *Experiments*)

> "One-Step Image Generation. We provide results for unconditional and class-conditional image generation on CIFAR-10 in Table 1 and Table 3, respectively." (Section 4.2, *Experiment Results*)

2. Number of trained model instances required to cover all tasks: 2 models

> "where  $\mathbf{x}$  is the desired ground truth image,  $G_{\theta}(\cdot)$  is unconditional ViT/GET with parameters  $\theta$ , and  $\mathbf{e}$  is the initial Gaussian noise. To train a class-conditional GET, we also use class labels  $\mathbf{y}$  in addition to noise/image pairs:" (Section 4.1, *Offline Distillation*)

> "where  $G_{\theta}^{c}(\cdot)$  is class-conditional ViT/GET with parameters  $\theta$ ." (Section 4.1, *Offline Distillation*)

3. Task–Model Ratio = (1) / (2)

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
