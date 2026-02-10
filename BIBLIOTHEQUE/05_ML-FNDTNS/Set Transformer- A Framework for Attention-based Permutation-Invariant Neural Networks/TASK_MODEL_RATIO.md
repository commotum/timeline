1. **Number of distinct tasks evaluated:** 5

"To evaluate the Set Transformer, we apply it to a suite of tasks involving sets of data points." (Section 5)

"#### 5.1. Toy Problem: Maximum Value Regression" (Section 5.1)
"#### 5.2. Counting Unique Characters" (Section 5.2)
"## 5.3. Amortized Clustering with Mixture of Gaussians" (Section 5.3)
"#### 5.4. Set Anomaly Detection" (Section 5.4)
"#### 5.5. Point Cloud Classification" (Section 5.5)

2. **Number of trained model instances required to cover all tasks:** 5

"Given a set of real numbers  $\{x_1, \ldots, x_n\}$ , the goal is to return  $\max(x_1, \dots, x_n)$ ." (Section 5.1)
"we train the model to predict the number of different characters inside the set." (Section 5.2)
"given a dataset X, we train a neural network to output parameters  $f(X;\lambda) = \{\pi(X), \{\mu_j(X), \sigma_j(X)\}_{i=1}^k\}$" (Section 5.3)
"The goal of this task is to find the image that does not belong to the set." (Section 5.4)
"We evaluated Set Transformers on a classification task using the ModelNet40 (Chang et al., 2015) dataset<sup>1</sup>, which contains three-dimensional objects in 40 different categories." (Section 5.5)

"$PMA_1(64,4) FC(1,-)$" (Supplementary Section 2.1, Table 1)
"PMA <sub>1</sub> (8,8)<br>FC(1, softplus)" (Supplementary Section 2.2, Table 3)
"$FC(4 \cdot (1 + 2 \cdot 2), -)$" (Supplementary Section 2.3.1, Table 4)
"PMA <sub>4</sub> (128, 4)<br>SAB(128, 4)<br>FC(256 · 8, -, -)" (Supplementary Section 2.4, Table 9)
"$\begin{array}{c} \operatorname{Dropout}(0.5) \\ \operatorname{PMA}_1(256,4) \\ \operatorname{Dropout}(0.5) \\ \operatorname{FC}(40,-) \end{array}$" (Supplementary Section 2.5, Table 10)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
