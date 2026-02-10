1. **Number of distinct tasks evaluated:** 10

   Verbatim evidence:
   - "Below we describe our results on all 10 tasks in more detail." (Section 4.4 RESULTS)
   - "1. Access Given a value k and an array A, return A[k]." (Section 4.2 Tasks)
   - "10. **WalkBST** Given a pointer to the root of a Binary Search Tree, and a path to be traversed (sequence of left/right steps), return the element at the end of the path." (Section 4.2 Tasks)

2. **Number of trained model instances required to cover all tasks:** 10

   Verbatim evidence:
   - "for every problem we selected a model that achieved error 0 during the training" (Section 4.4.1 EASY TASKS)
   - "For each of the tasks we have manually defined a sequence of subtasks with increasing difficulty" (Section B DETAILS OF CURRICULUM TRAINING)
   - "This category includes: **Permutation**, **ListK**, **ListSearch**, **Merge** and **WalkBST**. For all of them we had to perform an extensive random search to find a good set of hyperparameters." (Section 4.4.2 HARD TASKS)
   - Single jointly trained model covering all tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{10\ \text{tasks}}{10\ \text{models}} = 1
}
$$
