1. **Number of distinct tasks evaluated:** 4

- "A fast rollout policy  $p_\pi$  and supervised learning (SL) policy network  $p_\sigma$  are trained to predict human expert moves in a data set of positions." (Section: Figure 1 | Neural network training pipeline and architecture)
- "A reinforcement learning (RL) policy network  $p_\rho$  is initialized to the SL policy network, and is then improved by policy gradient learning to maximize the outcome (that is, winning more games) against previous versions of the policy network." (Section: Figure 1 | Neural network training pipeline and architecture)
- "Finally, a value network  $\nu_\theta$  is trained by regression to predict the expected outcome (that is, whether the current player wins) in positions from the self-play data set." (Section: Figure 1 | Neural network training pipeline and architecture)
- "To evaluate AlphaGo, we ran an internal tournament among variants of AlphaGo and several other Go programs, including the strongest commercial programs Crazy Stone<sup>13</sup> and Zen, and the strongest open source programs Pachi<sup>14</sup> and Fuego<sup>15</sup>." (Section: Evaluating the playing strength of AlphaGo)

2. **Number of trained model instances required to cover all tasks:** 4

- "We begin by training a supervised learning (SL) policy network  $p_{\sigma}$  directly from expert human moves." (Page: _page_1)
- "Similar to prior work <sup>13,15</sup>, we also train a fast policy  $p_{\pi}$  that can rapidly sample actions during rollouts." (Page: _page_1)
- "Next, we train a reinforcement learning (RL) policy network  $p_{\rho}$  that improves the SL policy network by optimizing the final outcome of games of self-play." (Page: _page_1)
- "Finally, we train a value network  $\nu_{\theta}$  that predicts the winner of games played by the RL policy network against itself." (Page: _page_1)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
