1. **Number of distinct tasks evaluated:** 6

- "We consider three matrix operations on  $20 \times 20$  matrices" (Section 4.1, Setup).
- "We run evaluations on two tasks: Sudoku solving and graph connectivity reasoning." (Section 4.2, Setup).
- "Test evaluation performance on the shortest path task." (Table 6, Planning Performance; see also Section 4.3, Planning).
- "**Extension to Visual Sudoku.** IRED can also be extended to deal with other input formats, such as images." (Section 4.2, Extension to Visual Sudoku).
- Visual Sudoku as a separate task count is **Not specified in the paper.**

2. **Number of trained model instances required to cover all tasks:** 6

- "**Continuous Tasks** We use dataset setups from (Du et al., 2022) for continuous tasks. Models were trained in approximately 2 hours" (Appendix A, Experimental Details).
- "For Sudoku, we train models for 50000 iterations" (Appendix A, Discrete Tasks).
- "For Connectivity tasks, we generate random graphs" and "We train models for 100000 iterations" (Appendix A, Discrete Tasks).
- "**Planning Task** For planning" and "We train models for 100000 iterations" (Appendix A, Planning Task).
- A single jointly trained model instance covering all evaluated tasks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
