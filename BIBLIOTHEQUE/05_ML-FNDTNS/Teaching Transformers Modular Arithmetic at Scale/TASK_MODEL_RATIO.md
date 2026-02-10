1. **Number of distinct tasks evaluated:** **4**

- "we train models to add N elements mod q" (Section 3.2 MODEL TRAINING AND EVALUATION).
- "We introduce a class of functions  $h:\mathbb{Z}_q^N\to\mathbb{Z}_q$  outside the aforementioned class" and Table 10 lists three specific functions: "$h_{j=1,k=1}$", "$h_{j=1,k=3}$", and "$h_{j=2,k=1}$" (Section 6 Beyond Modular Addition; Table 10).

2. **Number of trained model instances required to cover all tasks:** **4**

- "Following prior work, we train models to add N elements mod q (fixed N and q for each model)." (Section 3 METHODOLOGY).
- "We train models to predict outputs from these functions" (Section 6 Beyond Modular Addition).
- A single jointly trained model that performs all four tasks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
