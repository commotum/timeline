1. **Number of distinct tasks evaluated:** 160

> "As a benchmark, we use the subset of 160 object-centric ARC tasks introduced by Xu, Khalil, and Sanner (2023)." (Section: **Experiments**)

2. **Number of trained model instances required to cover all tasks:** 1

> "Figure 6 illustrates the pipeline sketch of GPAR, a two-stage system that employs GP to solve ARC tasks." (Section: **System Overview**)
>
> "For each ARC task, possible combinations are executed in order of increasing complexity, starting from lower values of n and v, fewer pointers, and simpler abstractions (e.g., 4-connected are considered before 8-connected abstractions) with a time limit of 1800s for each." (Section: **Parameters**)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{160\ \text{tasks}}{1\ \text{model}} = 160
}
$$
