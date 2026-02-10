1. **Number of distinct tasks evaluated:** 1

- "The goal is to search (as efficiently as possible) for the program that solves each task in the test set, given the provided DSL." (Section: The Problem)
- "Table 1 reports the success rate (as a percentage of solved tasks) of various approaches on the ARC-AGI evaluation set." (Section: Performance Comparison on ARC-AGI Eval Set)
- "10 new tasks are hand-crafted, specifically selected to guarantee that there is no structurally similar task ever generated in the training data." (Section: Generalization of LTS)

2. **Number of trained model instances required to cover all tasks:** 1

- "This is done by training three separate VLM instances, one on each of the three DSLs and their associated training data generators." (Section: GridCoder on Different DSLs)
- "This can be thought of as a curriculum progression, where the DSL version 3 model includes all primitives and task generators of DSL version 2, which itself includes all primitives and task generators of DSL version 1." (Section: GridCoder on Different DSLs)
- "as we grow the DSL from version 1 to version 3, the tasks that the approach was able to solve in DSL version 1 are still solved in subsequent sections." (Section: Scaling the DSL)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
