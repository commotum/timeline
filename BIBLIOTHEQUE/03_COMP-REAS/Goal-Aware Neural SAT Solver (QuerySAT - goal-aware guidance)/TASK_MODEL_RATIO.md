1. **Number of distinct tasks evaluated:** 5

> "We validate the theory in practice by building a neural SAT solver that we call QuerySAT and evaluate its performance on a wide range of SAT tasks - k-SAT, 3-SAT, 3-Clique, k-Coloring, and SHA-1 preimage attack." (Section IV. QUERYSAT)

> "Namely, we chose k-SAT, 3-SAT, and also 3-Clique, k-Coloring, and SHA-1 preimage attack problems represented as CNF Boolean formulas." (Section IV-C. Evaluation)

2. **Number of trained model instances required to cover all tasks:** 5 models

> "For all tasks, we generate a train set of 100k formulas and validation and test sets of 10k formulas each." (Section IV-C. Evaluation)

> "On the SHA-1 preimage attack, models are trained for 1M iterations, but on the rest of the tasks for 500k iterations." (Section IV-C. Evaluation)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
