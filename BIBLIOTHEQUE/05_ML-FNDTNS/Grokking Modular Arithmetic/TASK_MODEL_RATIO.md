1. **Number of distinct tasks evaluated:** 6

   Verbatim task evidence:
   - "$$f(n,m) = n + m \bmod p. (5)$$" (Section 3.1 Modular addition)
   - "•  $f(n,m) = n^2 + m^2 \mod p$ . Full solution is available and gives 100% accuracy." (Section C Some other modular functions)
   - "•  $f(n,m) = (n+m)^2 \mod p$ . The weights in the first layer are unmodified, while the weights in the second layer are given by" (Section C Some other modular functions)
   - "- f(n,m) = nm. We do not have an analytic solution. The activations are presented in Fig. 9" (Section C Some other modular functions)
   - "- $f(n,m)=n^2+m^2+nm \mod p$ . We do not have an analytic solution. This generalization on this function never reaches 100% unless most of the data is utilized,  $\alpha>0.95$ . See the learning curve in Fig. 10." (Section C Some other modular functions)
   - "- $f(n,m) = n^3 + nm^2 + m$ . We do not have an analytic solution. The generalization never rises above 1%. See the learning curve in Fig. 10." (Section C Some other modular functions)

2. **Number of trained model instances required to cover all tasks:** 6

   Verbatim evidence:
   - "Whether grokking happens or not depends on the modular function at hand assuming the architecture and optimizer are fixed." (Section 2 Set up and overview of results)
   - "The weights depend on the modular arithmetic task at hand." (Section 3.2 General modular functions and complexity)
   - Single jointly trained model covering all listed tasks without task-specific weights: "Not specified in the paper."

3. **Task-Model Ratio = (1) / (2):**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
