1. **Number of distinct tasks evaluated:** 4

   "We demonstrate the framework on four geometric optimization problems: sphere packing in hypercubes, circle packing maximizing sum of radii, the Heilbronn triangle problem, and star discrepancy minimization." (Abstract)

2. **Number of trained model instances required to cover all tasks:** 4

   "For each problem, we define a reward  $R: X \to \mathbb{R}$  aligned with the optimization objective." (Section 2.4, Reward-Guided Fine-Tuning)

   "The flow-matching model is trained for 200–500 epochs..." (Section 3.2, Sphere Packing in Hypercube, Training)

   "A conditional flow-matching model is then trained on the top portion of these SRP-refined configurations..." (Section 3.1, The Heilbronn Problem)

   "a flow-matching model on centers learns to propose new center configurations." (Section 3.1, Circle packing with maximal sum of radii)

   "A conditional flow-matching model (conditioned on  $(N, D^*)$ ) is then trained on the best half of the pushed samples..." (Section 3.1, Star discrepancy problem)

3. **Task–Model Ratio**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
