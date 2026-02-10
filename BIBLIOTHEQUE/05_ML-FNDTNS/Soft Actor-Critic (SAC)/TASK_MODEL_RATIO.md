1. **Number of distinct tasks evaluated:** 6

> "Table 2. SAC Environment Specific Parameters"
>
> "| Hopper-v1        | 3                 | 5            |"
> "| Walker2d-v1      | 6                 | 5            |"
> "| HalfCheetah-v1   | 6                 | 5            |"
> "| Ant-v1           | 8                 | 5            |"
> "| Humanoid-v1      | 17                | 20           |"
> "| Humanoid (rllab) | 21                | 10           |"

(Appendix D, "Table 2. SAC Environment Specific Parameters")

2. **Number of trained model instances required to cover all tasks:** 6

> "Table 2 lists the reward scale parameter that was tuned for each environment."

(Appendix D, "Hyperparameters")

> "The optimal reward scale varies between environments, and should be tuned for each task separately."

(Section 5.2, "Reward scale")

Single jointly trained multi-task model covering all tasks: Not specified in the paper.

3. **Task–Model Ratio**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
