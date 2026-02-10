1. **Number of distinct tasks evaluated:** 3

   - "To evaluate our algorithm, we consider three environments in the MuJoCo physics simulator." (Section 6.1 Environments)
   - "See Figure 1 for a visualization of the differences between expert and novice environments for the three tasks." (Section 6.1 Environments)
   - "Point: A pointmass attempts to reach a point in a plane." / "Reacher: A two DOF arm attempts to reach a designated point in the plane." / "Inverted Pendulum: A classic RL task wherein a pendulum must be made to balance via control." (Section 6.1 Environments)

2. **Number of trained model instances required to cover all tasks:** 3

   - "Initialize a novice policy \pi_{\theta}." (Algorithm 1, Section 6)
   - "Figure 3: Reward vs training iteration for reacher, inverted pendulum, and point environments." (Figure 3, page 7)
   - "Figure 6: Reward of final trained policy vs domain confusion weight  \lambda  for reacher, inverted pendulum, and point environments." (Figure 6, page 8)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
