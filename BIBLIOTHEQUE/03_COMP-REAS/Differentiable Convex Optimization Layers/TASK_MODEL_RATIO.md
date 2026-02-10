1. **Number of distinct tasks evaluated:** 2

- "In this section, we present two applications of differentiable convex optimization, meant to be suggestive of possible use cases for our layer. We give more examples in Appendix E." (Section "Examples" / §6)
- "#### 6.1 Data poisoning attack" (Section §6.1)
- "# 6.2 Convex approximate dynamic programming" (Section §6.2)

2. **Number of trained model instances required to cover all tasks:** 2

- Task 1 (Data poisoning attack) uses a separately fitted model: "We consider 30 training points and 30 test points in  $\mathbb{R}^2$ , and we fit a logistic model with elastic-net regularization." and "We used our convex optimization layer to fit this model and obtain the gradient of the test loss with respect to the training data." (Section §6.1, "Numerical example")
- Task 2 (Convex approximate dynamic programming) uses a separate policy model and optimization: "In this example, we take  $\mathcal{U}$  to be the unit ball and we represent  $\phi$  as a quadratic *control-Lyapunov* policy [74]." and "We can run stochastic gradient descent (SGD) on P, Q, and q to approximately solve (7), which requires differentiating through (8)." (Section §6.2, "ADP policy")

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
