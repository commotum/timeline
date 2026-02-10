1. **Number of distinct tasks evaluated:** 18

- "The first tasks we consider are eight simulated robotics tasks, implemented in MuJoCo (Todorov et al., 2012), and included in OpenAI Gym (Brockman et al., 2016)." (Section 3.1.1)
- "The second set of tasks we consider is a set of seven Atari games in the Arcade Learning Environment (Bellemare et al., 2013), the same games presented in Mnih et al., 2013." (Section 3.1.2)
- "we show that our algorithm can learn novel complex behaviors. We demonstrate:" (Section 3.2)
- "1. The Hopper robot performing a sequence of backflips (see Figure 4)." (Section 3.2)
- "2. The Half-Cheetah robot moving forward while standing on one leg." (Section 3.2)
- "3. Keeping alongside other cars in Enduro." (Section 3.2)

2. **Number of trained model instances required to cover all tasks:** 18

- "In our experiments, feedback is provided by contractors who are given a 1-2 sentence description of each task before being asked to compare several hundred to several thousand pairs of trajectory segments for that task" (Section 3.1)
- "This behavior was trained using 900 queries in less than an hour." (Section 3.2)
- "This behavior was trained using 800 queries in under an hour." (Section 3.2)
- "This was trained with roughly 1,300 queries and 4 million frames of interaction with the environment" (Section 3.2)
- Whether one single jointly trained model instance covers all tasks: "Not specified in the paper."

3. **Task-Model Ratio = (1) / (2)**

$$
\boxed{
\frac{18\ \text{tasks}}{18\ \text{models}} = 1
}
$$
