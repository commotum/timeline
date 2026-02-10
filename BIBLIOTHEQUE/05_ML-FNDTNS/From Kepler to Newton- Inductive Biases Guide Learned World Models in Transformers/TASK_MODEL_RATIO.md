1. **Number of distinct tasks evaluated: 2**

- Task 1 (Section 2.3): "To simplify the setting while retaining the essential features of tokenization, we adopt a 1D sine-wave dataset..." and "Transformer models are trained using next-token prediction with cross-entropy loss."
- Task 2 (Section 3): "To examine this question systematically, we introduce the Kepler dataset..." and "On this controlled testbed, we directly compare two formulations: next-state prediction (regression) and next-token prediction (classification)."
- The Kepler classification/regression setups are presented as "two formulations," not two different tasks (Section 3).

2. **Number of trained model instances required to cover all tasks: 2**

- One trained model instance for the 1D sine-wave task (Section 2.3: "Transformer models are trained using next-token prediction with cross-entropy loss.")
- One trained model instance for the Kepler trajectory-prediction task (Section 3: "we directly compare two formulations: next-state prediction (regression) and next-token prediction (classification).")
- A single jointly trained model covering both datasets/tasks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
