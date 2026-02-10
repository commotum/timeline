1. **Number of distinct tasks evaluated:** 4

   Verbatim evidence: "We introduce four competitive environments" and "Figure 1: Illustrations of competitive environments we consider in our work: Run to Goal, You Shall Not Pass, Sumo, and Kick and Defend." (Section 3, *Competitive Environments*)

2. **Number of trained model instances required to cover all tasks:** 6

   Verbatim evidence: "For the asymmetric games, you-shall-not-pass and kick-and-defend, we use separate policies for the two agents in a game." (Section 5.1, *Experimental Details*). For self-play, the paper states "We train agents on the Sumo task via self-play" (Section 5.4, *Effect of Opponent Sampling*) and "just a single policy is trained in a run" (Section 5.5.2, *Competing Against Ensemble of Policies*). It also includes run-to-goal in this self-play training discussion: "Fig. 2a shows the rewards during training with this naive approach (for the \"run to goal\" task with ant)." and "Note that for self-play this means that the policy at any time should be able to defeat random older versions of itself, thus ensuring continual learning." (Section 4.2, *Opponent Sampling*)

3. **Task-Model Ratio:**

$$
\boxed{
\frac{4\ \text{tasks}}{6\ \text{models}} = 0.67
}
$$
