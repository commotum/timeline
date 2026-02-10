1. **Number of distinct tasks evaluated:** 8.

Citation (Section 1, Introduction): "On the empirical side, we verify these predictions across 8 tasks in supervised learning, RL, and generative models, including ImageNet, CIFAR-10, SVHN, MNIST, BillionWord, Atari, OpenAI's Dota agent [BCD<sup>+</sup>18], and a variational autoencoder for images."

2. **Number of trained model instances required to cover all tasks:** 8 models.

Citations:
"For each of these tasks we demonstrate that the noise scale accurately predicts the largest usable batch size (at the order of magnitude level) and that gains to parallelism degrade in the manner predicted by theory." (Section 1, Introduction)

"- **SVHN** We train a simple CNN image classifier on the extended SVHN dataset [NWC<sup>+</sup>11]." (Section 3.2, Supervised Learning)

"- Language Modeling We train a single-layer LSTM for autoregressive prediction on the Billion Word dataset [CMS<sup>+</sup>13], and find good agreement between  $\mathcal{B}_{crit}$  and  $\mathcal{B}_{simple}$ ." (Section 3.2, Generative Modeling)

"- Atari We train RL agents with the policy gradient algorithm A2C [MBM+16] on seven Atari games [BNVB12] (Alien, Beamrider, Breakout, Pong, Qbert, Seaquest, Space Invaders)" and "- **Dota** The OpenAI Dota team has made it possible to train PPO [SWD<sup>+</sup>17] agents on both Dota 1v1 and 5v5 environments" (Section 3.2, Reinforcement Learning)

3. **Task–Model Ratio**

$$
\boxed{
\frac{8\ \text{tasks}}{8\ \text{models}} = 1
}
$$
