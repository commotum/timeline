1. **Number of distinct tasks evaluated:** 2

- "Figure 1: Illustrative example using the two-room gridworld (shown inset) contrasting planning with reward-respecting and shortest-path options for reaching the bottleneck state." (Section 1, Figure 1 caption)
- "The larger problem used in these comparisons is the four-room episodic gridworld depicted in each of the four parts of Figure 6, with a start state in the upper-left room (highlighted in green) and a terminal goal state in the lower-right room." (Section 7)

2. **Number of trained model instances required to cover all tasks:** 2

- "For the illustrative example in Figure 1, we learned models of the four options corresponding to actions and of the one reward-respecting option for attaining the hallway state." (Section 4)
- "For each of these four ways of producing options, we learn models of their options and use the models for planning exactly as described earlier in this paper." (Section 7)
- A single jointly trained model instance spanning both gridworld tasks is Not specified in the paper.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
