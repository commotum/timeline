1. **Number of distinct tasks evaluated:** 3

"Separate experiments are conducted for regression and classification tasks." (Section II.C.1.b, Experiments)

"Estimation of the conditional probability of a complete sequence of classes of length T using all available input information [i.e., compute  $\Pr(c_1, c_2, \dots, c_T | \mathbf{x}_1^T)$ ]. In this case, the outputs are treated as being statistically dependent, which makes the estimation more difficult and requires a slightly different network structure than the one used in the first part." (Section I.C, Organization of the Paper)

2. **Number of trained model instances required to cover all tasks:** 3

"Separate experiments are conducted for regression and classification tasks." (Section II.C.1.b, Experiments)

"...requires a slightly different network structure than the one used in the first part." (Section I.C, Organization of the Paper)

"Two different structures of the modified BRNN (one for the forward and the other for the backward posterior probability) are trained separately as classifiers using the cross-entropy objective function." (Section III.C.1, Experiments)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
