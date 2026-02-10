1. **Number of distinct tasks evaluated:** 87
   - "DMLab-30 (a set of 30 tasks from the DeepMind Lab environment (Beattie et al., 2016)) and Atari-57 (all available Atari games in Arcade Learning Environment (Bellemare et al., 2013a))." (Abstract)
   - "Its 57 tasks pose challenging reinforcement learning problems including exploration, planning, reactive play and complex visual input." (Section 5.3.2, Atari)

2. **Number of trained model instances required to cover all tasks:** 2
   - "For multi-task learning we train agents—each with one set of weights for all tasks—on a newly introduced collection of 30 DeepMind Lab tasks and on all 57 games of the Atari Learning Environment (Bellemare et al., 2013a))." (Section 5, Experiments)
   - "In addition to individual per-game experts, trained for 200 million frames with a fixed set of hyperparameters, we train an IMPALA Atari-57 agent—one agent, one set of weights—on all 57 Atari games at once for 200 million frames per game or a total of 11.4 billion frames." (Section 5.3.2, Atari)
   - Single model jointly covering both DMLab-30 and Atari-57: Not specified in the paper.

3. **Task–Model Ratio**

$$
\boxed{
\frac{87\ \text{tasks}}{2\ \text{models}} = 43.5
}
$$
