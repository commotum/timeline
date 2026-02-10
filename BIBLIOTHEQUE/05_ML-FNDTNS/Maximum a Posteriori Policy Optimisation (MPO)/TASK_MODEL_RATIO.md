1. **Number of distinct tasks evaluated:** 25

- “The suite of continuous control tasks that we are evaluating against contains 18 tasks, comprising a wide range of domains including well known tasks from the literature.” (Section 5.1, `EVALUATION ON CONTROL SUITE`)
- “Figure 3: MPO on high-dimensional control problems (Parkour Walker2D and Humanoid walking from control suite).” (Figure 3 caption, Section 5.2)
- “In addition to the walker experiment we have also evaluated MPO on the Parkour domain using a humanoid body (with 22 degrees of freedom) which was learned successfully (not shown in the plot, please see the supplementary video).” (Section 5.2, `HIGH-DIMENSIONAL CONTINUOUS CONTROL`)
- “Table 1: Results on a subset of the ALE environments in comparison to baselines taken from (Bellemare et al., 2017)” (Table 1, Appendix B)
- “| Pong       | 14.6     | 19.5     | 20.9                  | 20.9       | 20.9     |” (Table 1, Appendix B)
- “| Breakout   | 30.5     | 385.5    | 366.0                 | <b>748</b> | 360.5    |” (Table 1, Appendix B)
- “| Q*bert     | 13,455.0 | 13,117.3 | 18,760.3              | 23,784     | 10,317.0 |” (Table 1, Appendix B)
- “| Tennis     | -8.3     | 12.2     | 0.0                   | 23.1       | 22.2     |” (Table 1, Appendix B)
- “| Boxing     | 12.1     | 88.0     | 98.9                  | 97.8       | 82.0     |” (Table 1, Appendix B)

2. **Number of trained model instances required to cover all tasks:** 25 models

- “The results for MPO (non-parameteric) – and a comparison to an implementation of state-of-the-art algorithms from the literature in our framework – on all the environments from the control suite that we tested on are shown in Figure 4.” (Section 5.1.2, `Complete results on the control suite`)
- “For our experiments we evaluate our MPO algorithm across a wide range of tasks. Specifically, we start by looking at the continuous control tasks of the DeepMind Control Suite (Tassa et al. (2018), see Figure 1), and then consider the challenging parkour environments recently published in Heess et al. (2017). In both cases we use a Gaussian distribution for the policy whose mean and covariance are parameterized by a neural network (see appendix for details). In addition, we present initial experiments for discrete control using ATARI environments using a categorical policy distribution (whose logits are again parameterized by a neural network) in the appendix.” (Section 5, `EXPERIMENTS`)
- “As a proof of concept – showcasing the robustness of our algorithm and its hyperparameters – we performed an experiment on a subset of the games contained contained in the "Arcade Learning Environment" (ALE). For this experiment we used *the same hyperparameter* settings for the KL constraints as for the continuous control experiments as well as the same learning rate and merely altered the network architecture to the standard network structure used by DQN Mnih et al. (2015) – and created a seperate network with the same architecture, but predicting the parameters of the policy distribution.” (Appendix B, `ADDITIONAL EXPERIMENT: DISCRETE CONTROL`)
- Single jointly trained model across all tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{25\ \text{tasks}}{25\ \text{models}} = 1
}
$$
