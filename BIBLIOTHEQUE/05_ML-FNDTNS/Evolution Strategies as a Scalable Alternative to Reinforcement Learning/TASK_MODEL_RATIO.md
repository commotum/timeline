1. **Number of distinct tasks evaluated:** 58

   - "We ran our parallel implementation of Evolution Strategies, described in Algorithm 2, on 51 Atari 2600 games available in OpenAI Gym [Brockman et al., 2016]." (Section 4.2, Atari)
   - Table 1 lists six MuJoCo benchmark environments: "HalfCheetah", "Hopper", "InvertedDoublePendulum", "InvertedPendulum", "Swimmer", "Walker2d". (Table 1, Section 4.1, Mu.JoCo)
   - "We picked the 3D Humanoid walking task from OpenAI Gym [Brockman et al., 2016] as the test problem for our scaling experiment, because it is one of the most challenging continuous control problems solvable by state-of-the-art RL techniques, which require about a day to learn on modern hardware [Schulman et al., 2015, Duan et al., 2016a]." (Section 4.3, Parallelization)
   - Total: 51 + 6 + 1 = 58.

2. **Number of trained model instances required to cover all tasks:** 58

   - Atari training is task-specific: "All games were trained for 1 billion frames" and "we can bring down the time required for the training process to about one hour per game." (Section 4.2, Atari)
   - MuJoCo evaluation is task/environment-specific: "We found that ES was able to solve these tasks up to TRPO's final performance after 5 million timesteps of environment interaction." and results are listed by environment in Table 1. (Section 4.1, Mu.JoCo)
   - 3D Humanoid is evaluated as a separate task in Section 4.3.
   - Total required trained instances for full task coverage: 51 + 6 + 1 = 58.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{58\ \text{tasks}}{58\ \text{models}} = 1
}
$$
