1. **Number of distinct tasks evaluated:** **800 tasks**

   - Verbatim quote: "Each puzzle in the benchmark consists of a different hidden rule, which the system must apply to an input colored grid to produce a ground truth target colored grid." (Section **2 BACKGROUND: THE ARC-AGI BENCHMARK**)
   - Verbatim quote: "There are 400 training puzzles and they are easier to solve than the 400 evaluation puzzles." (Section **L ADDITIONAL DETAILS ABOUT THE ARC-AGI BENCHMARK**)

2. **Number of trained model instances required to cover all tasks:** **800 models**

   - Verbatim quote: "foreach puzzle P in ARC-AGI dataset do" and "Randomly initialize weights \theta for equivariant_NN_{\theta};" (Section **3.2 SEED OPTIMIZATION**, **Algorithm 3: CompressARC**)
   - Verbatim quote: "Template Algorithm 1 includes a hard-coded value of  $\theta$  for every single puzzle." (Section **K.1 JOINT COMPRESSION VIA WEIGHT SHARING BETWEEN PUZZLES**)

3. **Task–Model Ratio**

$$
\boxed{
\frac{800\ \text{tasks}}{800\ \text{models}} = 1
}
$$
