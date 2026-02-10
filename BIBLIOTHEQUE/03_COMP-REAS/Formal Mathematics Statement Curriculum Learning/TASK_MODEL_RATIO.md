1. **Number of distinct tasks evaluated:** **1**

"Our expert iteration process takes as input: (i) a set of formal statements St, (ii) a function  $a:St\to\mathbb{N}$  indicating the number of proof search attempts to run per statement at each iteration..." (Section 4.5, *Iterated sampling and training*)

"In this section we propose to set St to the union of the statements in mathlib-train, synth-ineq and miniF2F-curriculum." (Section 6.2, *Transfer to miniF2F*)

2. **Number of trained model instances required to cover all tasks:** **1**

"Throughout this paper we focus on a model with 36 layers and 774 million trainable parameters..." (Section 4.1, *Model*)

"We'll denote the fully iterated model from this section as  $\theta_9^{full}$ ." (Section 6.2, *Transfer to miniF2F*)

"Table 2. Performance of  $\theta_1$  (value-function based search),  $\theta_9^{mathlib}$  (expert iterated on mathlib-train) and  $\theta_9^{full}$  (expert iterated on our full curriculum) on mathlib-{valid, test} and miniF2F-{valid, test}." (Section 6.3, *Results*)

3. **Task–Model Ratio**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
