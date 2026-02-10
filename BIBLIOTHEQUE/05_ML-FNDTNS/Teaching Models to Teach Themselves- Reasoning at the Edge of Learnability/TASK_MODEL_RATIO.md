1. **Number of distinct tasks evaluated:** 3

   Verbatim evidence: "We use three such benchmarks: MATH (Hendrycks et al., 2021), HARP (Yue et al., 2024), and OlympiadBench (He et al., 2024)." (Section 4.1)

2. **Number of trained model instances required to cover all tasks:** 2 models

   Verbatim evidence: "We train with SOAR on MATH and HARP, keeping OlympiadBench held-out to test cross-dataset generalization." (Section 4.2)

   Verbatim evidence: "Figure 4 shows that synthetic questions from PQ-MATH, PQ-HARP, and Intrinsic-T transfer to OlympiadBench, an OOD dataset (+6% and +3% respectively over Hard-Only)." (Section 5.1)

   A single jointly trained multi-task SOAR model covering all three benchmarks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{2\ \text{models}} = 1.5
}
$$
