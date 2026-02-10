1. **Number of distinct tasks evaluated:** 2. Evidence: "We evaluate our meta-learning agents along with a number of baselines on a (single-agent) locomotion task with handcrafted nonstationarity and on iterated adaptation games in RoboSumo." (Section 1, Introduction). "We have designed a set of environments for testing different aspects of continuous adaptation methods in two scenarios: (i) simple environments that change from episode to episode according to some underlying dynamics, and (ii) a competitive multi-agent environment, RoboSumo, that allows different agents to play sequences of games against each other and keep adapting to incremental changes in each other's policies." (Section 4, ENVIRONMENTS).

2. **Number of trained model instances required to cover all tasks:** 2 models. Evidence: "Training in nonstationary locomotion. We train all methods on the same collection of nonstationary locomotion environments constructed by choosing all possible pairs of legs whose joint torques are scaled except 3 pairs that are held out for testing (i.e., 12 training and 3 testing environments for the six-leg creature)." (Section 5.1, The setup). "**Training in RoboSumo.** To ensure consistency of the training curriculum for all agents, we first pre-train a number of policies of each type for every agent type via pure self-play with the PPO algorithm (Schulman et al., 2017; Bansal et al., 2018)." and "Next, we train the baselines and the meta-learning agents against the pool of pre-trained opponents<sup>7</sup> concurrently." (Section 5.1, The setup).

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
