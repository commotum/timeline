1. **Number of distinct tasks evaluated: 4**

> "We evaluated our approach on the classic cart-pole balancing problem, as well as several challenging 3D locomotion tasks: (1) bipedal locomotion; (2) quadrupedal locomotion; (3) dynamically standing up, for the biped, which starts off laying on its back." (Section 6.2 EXPERIMENTAL SETUP)

2. **Number of trained model instances required to cover all tasks: 4**

> "We used the same neural network architecture for all of the 3D robot tasks, which was a feedforward network with three hidden layers, with 100, 50 and 25 tanh units respectively." (Section 6.2.1 ARCHITECTURE)

> "For the simpler cartpole task, we used a linear policy, and a neural network with one 20-unit hidden layer as the value function." (Section 6.2.1 ARCHITECTURE)

> "The humanoid model has 33 state dimensions and 10 actuated degrees of freedom, while the quadruped model has 29 state dimensions and 8 actuated degrees of freedom." (Section 6.2.2 TASK DETAILS)

Whether a single jointly trained model instance was used across all tasks is **Not specified in the paper.** Based on the task-specific setups reported above, covering all distinct tasks requires separate trained instances per task.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
