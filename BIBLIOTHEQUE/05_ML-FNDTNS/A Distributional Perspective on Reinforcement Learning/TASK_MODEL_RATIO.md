1. **Number of distinct tasks evaluated:** 58

   - "We used five training games (Fig 3) and 52 testing games." (Section 5. Evaluation on Atari 2600 Games)
   - "In this section, we empirically validate these results on the CliffWalk domain shown in Figure 11." (Section D. Comparison of Sampled Wasserstein Loss and Categorical Projection)

2. **Number of trained model instances required to cover all tasks:** 58

   - "For our study, we use the DQN architecture (Mnih et al., 2015), but output the atom probabilities  $p_i(x,a)$  instead of action-values, and chose  $V_{\text{MAX}} = -V_{\text{MIN}} = 10$  from preliminary experiments over the training games. We call the resulting architecture  $Categorical\ DQN$ . We replace the squared loss  $(r + \gamma Q(x', \pi(x')) - Q(x,a))^2$  by  $\mathcal{L}_{x,a}(\theta)$  and train the network to minimize this loss." (Section 5. Evaluation on Atari 2600 Games)
   - "Figure 7. Percentage improvement, per-game, of C51 over Double DQN, computed using van Hasselt et al.'s method." (Figure 7)

3. **Task–Model Ratio:** 58 / 58 = 1

$$
\boxed{
\frac{58\ \text{tasks}}{58\ \text{models}} = 1
}
$$
