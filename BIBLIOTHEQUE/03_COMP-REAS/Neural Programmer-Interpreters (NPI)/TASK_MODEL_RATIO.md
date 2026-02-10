1. **Number of distinct tasks evaluated: 4**

- "Table 1 shows the results for addition, sorting and canonicalizing 3D car models." (Section 4.4, "SOLVING MULTIPLE TASKS WITH A SINGLE NETWORK")
- "Table 1 shows the result of adding a maximum-finding program MAX to a multitask NPI trained on addition, sorting and canonicalization." (Section 4.3, "Learning New Programs with a fixed core")
- "| Maximum         | -      | -     | 100.0 |" (Table 1, Section 4.4)

2. **Number of trained model instances required to cover all tasks: 2**

- "As shown in Table 1, one multi-task NPI can learn all three programs (and necessarily the 21 subprograms) with comparable accuracy compared to each single-task NPI." (Section 4.4)
- "During training we froze all weights except for the two newly-added program embeddings." (Section 4.3)
- "after training a single multi-task model as outlined in the following section, learning the MAX program with this fixed-core multi-task NPI results in no performance deterioration for all three tasks." (Section 4.3)
- Whether all four tasks were jointly trained from scratch in one single training run: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
