1. **Number of distinct tasks evaluated:** 1

"Even though our model is quite general, in this paper, we apply Neural Programmer to the task of question answering on tables, a task that has not been previously attempted by neural networks." (Section 2 NEURAL PROGRAMMER)

"In the following, we benchmark the performance of Neural Programmer on various versions of the table-comprehension dataset. We slowly increase the difficulty of the task by changing the table properties (more columns, mixed numeric and text entries) and question properties (word variability)." (Section 3.2 Models)

2. **Number of trained model instances required to cover all tasks:** 1

"Neural Programmer currently supports two types of outputs: a) a scalar output, and b) a list of items selected from the table (i.e., table lookup)." (Section 2.3 OPERATIONS)

"During training, depending on whether the answer is a scalar or a lookup from the table we have two different loss functions." (Section 2.5 Training Objective)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{tasks}}{1\ \text{model}} = 1
}
$$
