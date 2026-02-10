1. **Number of distinct tasks evaluated:** 8.

   Verbatim evidence:
   - Section 5 ("**Tasks.**"): "Our experiments are on a series of algorithmic tasks shown in Table 1a. The COPY, REVERSE, and BIGRAM FLIP tasks are based on Grefenstette et al. (2015); the DOUBLE and INTERLEAVED ADD tasks are designed in a similar vein. Additionally we also include three harder tasks: ODD FIRST, REPEAT COPY, and PRIORITY SORT."
   - Table 1a (Section 5): "1 - COPY", "2 - Reverse", "3 - BIGRAM FLIP", "4 - Double", "5 - Interleaved Add", "6 - Odd First", "7 - Repeat Copy", "8 - Priority Sort".

2. **Number of trained model instances required to cover all tasks:** 8 models.

   Verbatim evidence:
   - Section 5 ("**Tasks.**"): "In ODD FIRST, the model must output the odd-indexed elements first, followed by the even-indexed elements. In REPEAT COPY, each model must repeat a sequence of length 20, N times. In PRIORITY SORT, each item of the input sequence is given a priority, and the model must output them in priority order."
   - Section 5 ("**Model Setup.**"): "For all tasks, the LSTM baseline has 1 to 4 layers, each with 256 cells."
   - Section 5: "We train each model in two regimes, one with a small number of samples (16K) and one with a large number of samples (320K)."
   - Section 5 ("**Results.**"): "Main results comparing the different memory systems and read computations on a series of tasks are shown in Table 1b."
   - Table 1a (Section 5), "$ \mathcal{V} $" entries include "10" (for "4 - Double" and "5 - Interleaved Add") and "128" (for "1 - COPY", "2 - Reverse", and others).
   - Whether a single jointly trained model is explicitly reported for all tasks: "Not specified in the paper."
   - Inference from the quoted setup/results text: task-wise training/evaluation is reported, and no single jointly trained multi-task model is described; therefore, covering all 8 tasks requires 8 separately trained task models.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{8\ \text{tasks}}{8\ \text{models}} = 1
}
$$
