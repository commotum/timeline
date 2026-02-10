1. **Number of distinct tasks evaluated:** 5

   Verbatim evidence (Section **II.C. Results Summary and Comparison with Prior Art**, Table I):
   - "QTM, 100, Ours" (2x2x2 block)
   - "QTM, 1000, [3]" (3x3x3 block)
   - "UQTM, 82, [34]" (3x3x3 block)
   - "UQTM, 43, [34]" (4x4x4 block)
   - "UQTM, 19, [34]" (5x5x5 block)

   Verbatim evidence that QTM and UQTM are distinct settings (Section **II.C**, Table I footnote): "The 2023 Kaggle Santa Challange dataset uses modified QTM with unfixed corners and centers of the cube, which is marked UQTM."

2. **Number of trained model instances required to cover all tasks:** 73

   Verbatim definition of model-instance counting (Section **II.A. Proposed Machine Learning Approach**, "Multi-agency"): "We call each trained neural network an agent."

   Verbatim parameter definition (Section **II.B. Optimality vs the Proposed Approach Parameters**): "A – the number of agents"

   Verbatim task-wise agent counts from Section **II.C**, Table I (ours):
   - 2x2x2, "QTM, 100, Ours" -> "1"
   - 3x3x3, "QTM, 1000, [3]" -> "1"
   - 3x3x3, "UQTM, 82, [34]" -> "1"
   - 4x4x4, "UQTM, 43, [34]" -> "1"
   - 5x5x5, "UQTM, 19, [34]" -> "69"

   Sum: 1 + 1 + 1 + 1 + 69 = 73.

   Verbatim coverage statement (Reference **[51]**): "All the solvers presented in the Table I managed to solve all the scrambles from the listed datasets."

3. **Task–Model Ratio = (1) / (2):** 5 / 73

$$
\boxed{
\frac{5\ \text{tasks}}{73\ \text{models}} = 0.0685
}
$$
