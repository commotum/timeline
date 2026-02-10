1. **Number of distinct tasks evaluated:** 3

- “We introduce the simpler *Pattern* task to investigate LPN's dynamics before large-scale ARC-AGI training, see Section A.” (Section 5.1 Setup)
- “### 5.3 String Manipulation Task” (Section 5.3)
- “### 5.7 ARC-AGI 2024” (Section 5.7)

2. **Number of trained model instances required to cover all tasks:** 3

- Pattern-task training is reported separately: “For each training method, we train a small 1M-parameter model for 20k steps with a batch size of 128 and evaluate it with different inference modes.” (Section 5.2 Pattern Task)
- String-task evaluation is reported as a separate experiment: “Table 2: Exact match accuracy (%) on the test set for the string manipulation task.” (Section 5.3 String Manipulation Task)
- ARC-AGI training is reported separately: “We train a 178M-parameter LPN with a 256-dim latent space for 100k steps for 2 days on a TPU v4-32, see Section E.” (Section 5.7 ARC-AGI 2024)
- A single jointly trained model covering all three task families is: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
