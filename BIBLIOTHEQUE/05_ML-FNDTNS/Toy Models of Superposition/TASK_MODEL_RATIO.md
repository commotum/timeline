1. **Number of distinct tasks evaluated:** 2

   - "Our goal is to explore whether a neural network can project a high dimensional vector  $x \in R^n$  into a lower dimensional vector  $h \in R^m$  and then recover it." (Section: **Experiment Setup** in **Demonstrating Superposition**)
   - "Specifically, we'll have the model compute  $y=\mathrm{abs}(x)$ ." (Section: **Computation in Superposition**)

2. **Number of trained model instances required to cover all tasks:** 2

   - "To explore this, we consider a new setup where we imagine our input and output layer to be the layers of our hypothetical disentangled model, but have our hidden layer be a smaller layer we're imagining to be the observed model which might use superposition. We'll then try to compute a simple non-linear function and explore whether it can use superposition to do this." (Section: **Computation in Superposition**)
   - "Following the previous section, we'll consider the \"ReLU hidden layer\" toy model variant, but no longer tie the two weights to be identical" (Section: **Experiment Setup** in **Computation in Superposition**)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
