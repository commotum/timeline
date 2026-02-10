1. **Number of distinct tasks evaluated:** 59

   - "Namely, we used 7 simulated robotics tasks<sup>2</sup> implemented in OpenAI Gym [Bro+16], which use the MuJoCo [TET12] physics engine." (Section 6.1, *Comparison of Surrogate Objectives*)
   - "The three tasks we test on are (1) RoboschoolHumanoid: forward locomotion only, (2) RoboschoolHumanoidFlagrun: position of target is randomly varied every 200 timesteps or whenever the goal is reached, (3) RoboschoolHumanoid-FlagrunHarder, where the robot is pelted by cubes and needs to get up off the ground." (Section 6.3, *Showcase in the Continuous Domain: Humanoid Running and Steering*)
   - "A table of results and learning curves for all 49 games is provided in Appendix B." (Section 6.4, *Comparison to Other Algorithms on the Atari Domain*)
   - "Here we include a comparison of PPO against A2C on a larger collection of 49 Atari games." (Appendix B, *Performance on More Atari Games*)
   - Count used: \(7 + 3 + 49 = 59\).

2. **Number of trained model instances required to cover all tasks:** 59

   - "We do one million timesteps of training on each one." (Section 6.1, *Comparison of Surrogate Objectives*)
   - "Each algorithm was run on all 7 environments, with 3 random seeds on each." (Section 6.1, *Comparison of Surrogate Objectives*)
   - "See Figure 5 for still frames of a learned policy, and Figure 4 for learning curves on the three tasks." (Section 6.3, *Showcase in the Continuous Domain: Humanoid Running and Steering*)
   - "Table 2 shows the number of games \"won\" by each algorithm, where we compute the victor by averaging the scoring metric across three trials." (Section 6.4, *Comparison to Other Algorithms on the Atari Domain*)
   - Whether one jointly trained model instance covers all tasks: Not specified in the paper.
   - Using the reported per-task/per-game training setup above, one trained instance per evaluated task gives 59 models.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{59\ \text{tasks}}{59\ \text{models}} = 1
}
$$
