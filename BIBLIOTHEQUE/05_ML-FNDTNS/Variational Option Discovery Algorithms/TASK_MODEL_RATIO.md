1. **Number of distinct tasks evaluated: 7**

"**Test environments**: Our core comparison experiments is on a slate of locomotion environments: a custom 2D point agent, the HalfCheetah and Swimmer robots from the OpenAI Gym [Brockman et al., 2016], and a customized version of Ant from Gym where contact forces are omitted from the observations. We also tried running variational option discovery on two other interesting simulated robots: a dextrous hand (with  $S \in \mathbb{R}^{48}$  and  $A \in \mathbb{R}^{20}$ , based on Plappert et al. [2018]), and a new complex humanoid environment we call 'toddler' (with  $S \in \mathbb{R}^{335}$  and  $A \in \mathbb{R}^{35}$ ). Lastly, we investigated applicability to downstream tasks in a modified version of Ant-Maze (Frans et al. [2018])." (Section 4, "Test environments")

2. **Number of trained model instances required to cover all tasks: 7**

"Each single seed corresponds to a single policy with K=64 behaviors." (Section D.1)

"Downstream Tasks: We investigated whether behaviors learned by variational option discovery could be used for a downstream task by taking a policy trained with VALOR on the Ant robot (Uniform distribution, seed 10; see Appendix D.7), and using it as the lower level of a two-level hierarchical policy in Ant-Maze. We held the VALOR policy fixed throughout downstream training, and only trained the upper level policy, using A2C as the RL algorithm (with reinforcement occuring only at the lower level—the upper level actions were trained by signals backpropagated through the lower level)." (Section 5, "Downstream Tasks")

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{7\ \text{tasks}}{7\ \text{models}} = 1
}
$$
