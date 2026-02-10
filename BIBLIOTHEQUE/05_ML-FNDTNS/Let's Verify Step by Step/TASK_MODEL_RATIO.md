1. **Number of distinct tasks evaluated:** 1

   "Our process-supervised model solves 78% of problems from a representative subset of the MATH test set." (Abstract)

   "To get some measure of out-of-distribution generalization, we evaluate our large-scale ORM and PRM on a held-out set of 224 STEM questions" (Section 5 OOD Generalization)

2. **Number of trained model instances required to cover all tasks:** 2

   "At each model scale, we use a single fixed model to generate all solutions. We call this model the *generator*." (Section 2.1 Scope)

   "We evaluate a reward model by its ability to perform best-of-N search over uniformly sampled solutions from the generator." (Section 2.1 Scope)

3. **Task–Model Ratio:**

$$
\boxed{
\frac{1\ \text{tasks}}{2\ \text{models}} = 0.5
}
$$
