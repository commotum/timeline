1. **Number of distinct tasks evaluated:** 100

   - "The Abstraction and Reasoning Corpus (ARC) dataset  $\mathcal{D}$  consists of a collection of tasks (also called riddles in this paper)" (Section 2.1).
   - "The ARC dataset consists of 400 training riddles, 400 public evaluation riddles, and 100 private evaluation riddles" and "We report our results on the private test set" (Sections 4.1 and 4.1.1).

2. **Number of trained model instances required to cover all tasks:** 100

   - "During evaluation, we leverage each test riddle's demonstration examples to create synthetic training data" (Section 3.2).
   - "We then perform a brief round of fine-tuning on these augmented riddles before generating predictions for the test grids" (Section 3.2).

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{100\ \text{tasks}}{100\ \text{models}} = 1
}
$$
