1. **Number of distinct tasks evaluated:** 1

- "We evaluate on the binary cumulative parity task, a canonical benchmark for testing long-range dependency modeling [42]." (Section 5.1.1, *Task: Binary Cumulative Parity*)

2. **Number of trained model instances required to cover all tasks:** 1

- "We compare two architectures with matched capacity:" (Section 5.1.2, *Models and Training*)
- "Equilibrium Transformer (EqT): Identical base architecture augmented with the Equilibrium Refinement Module (Section 3). Energy function includes reverse prediction ( $\mathcal{L}_{rev}$ ), masked reconstruction ( $\mathcal{L}_{mask}$ ), prediction confidence ( $\mathcal{L}_{conf}$ ), and proximal regularization. Total parameters:  $\sim$ 6.8M (+8% overhead). Training uses K=2 refinement iterations for gradient stability; evaluation uses  $K\in\{8,32\}$ ." (Section 5.1.2, *Models and Training*)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
