1. **Number of distinct tasks evaluated:** 5.

   - "We show that \"model-wise double-descent\" occurs for various modern datasets (CIFAR-10, CIFAR-100, IWSLT'14 de-en, with varying amounts of label noise), model architectures (CNNs, ResNets, Transformers), optimizers (SGD, Adam), number of train samples, and training procedures (data-augmentation, and regularization)." (Section 2, *Our results*).
   - "Figure 8: Transformers on language translation tasks: Multi-head-attention encoder-decoder Transformer model trained for 80k gradient steps with labeled smoothed cross-entropy loss on IWSLT'14 Germanto-English (160K sentences) and WMT'14 English-to-French (subsampled to 200K sentences) dataset." (Section 5, Figure 8).
   - "Figure 14: **Random Fourier Features** on the Fashion MNIST dataset." (Appendix D, Figure 14).

2. **Number of trained model instances required to cover all tasks:** 5.

   - "We consider three families of architectures: ResNets, standard CNNs, and Transformers." (Section 4, *EXPERIMENTAL SETUP*).
   - "For ResNets and CNNs, we train with cross-entropy loss, and the following optimizers: (1) Adam with learning-rate 0.0001 for 4K epochs; (2) SGD with learning rate  $\propto \frac{1}{\sqrt{T}}$  for 500K gradient steps. We train Transformers for 80K gradient steps, with 10% label smoothing and no drop-out." (Section 4, *EXPERIMENTAL SETUP*).
   - A single jointly trained model instance that covers all listed tasks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
