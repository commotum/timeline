1. **Number of distinct tasks evaluated:** 10
"Ten popular Atari games were selected and fixed before starting this research." (Section 4.1, Training and Evaluation)

2. **Number of trained model instances required to cover all tasks:** 10 models
"Since different tasks often have different action sets, a separate output layer (called the controller layer) is trained for each task and the id of the task is used to switch to the correct output during both training and evaluation." (Section 3.3, Multi-Task Policy Distillation)
"The multi-task networks had a separate MLP output (controller) layer for each task." (Section 4.1, Training and Evaluation)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{10\ \text{tasks}}{10\ \text{models}} = 1
}
$$
