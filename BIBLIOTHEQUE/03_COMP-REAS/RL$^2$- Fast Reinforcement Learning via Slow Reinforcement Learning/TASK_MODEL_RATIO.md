1. **Number of distinct tasks evaluated:** 3
   - [Section 3 EVALUATION] "For the first question, we evaluate  $RL^2$  on two sets of tasks, multi-armed bandits (MAB) and tabular MDPs."
   - [Section 3 EVALUATION] "For the second question, we evaluate  $RL^2$  on a vision-based navigation task."

2. **Number of trained model instances required to cover all tasks:** 3
   - [Section A DETAILED EXPERIMENT SETUP] "embed the states and actions as described separately below for each experiments."
   - [Section A.1 MULTI-ARMED BANDITS] "we use a constant embedding 0 as a placeholder in place of the states, and a one-hot embedding for the actions."
   - [Section A.2 TABULAR MDPS] "We use a one-hot embedding for the states and actions separately, which are then concatenated together."
   - [Section A.3 VISUAL NAVIGATION] "For this task, we use a neural network to form the joint embedding."
   - Not specified in the paper.

3. **Task–Model Ratio**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
