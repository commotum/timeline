1. **Number of distinct tasks evaluated: 24**

- "#### 3.1 CLEVR" (Section 3.1 CLEVR)
- "#### 3.2 Sort-of-CLEVR" (Section 3.2 Sort-of-CLEVR)
- "There are 20 tasks, each corresponding to a particular type of reasoning, such as deduction, induction, or counting." (Section 3.3 bAbI)
- "We defined two separate tasks: 1) infer the existence or absence of connections between balls when only observing their color and coordinate positions across multiple sequential frames, and 2) count the number of systems on the table-top, again when only observing each ball's color and coordinate position across multiple sequential frames." (Section 3.4 Dynamic physical systems)

2. **Number of trained model instances required to cover all tasks: 5**

- "Our model was trained on the joint version of bAbI (all 20 tasks simultaneously), using the full dataset of 10K examples per task." (Section 4 Models)
- "The same model was used for the counting task, but this time the output layer of the RN was a linear layer with 10 units." (Section F Dynamic physical system reasoning)
- "For the CLEVR-from-pixels task we used:" (Section 4 Models)
- "In this task our model used:" (Section D Sort-of-CLEVR)
- "When we trained our models, we used either the pixel version or the state description version, depending on the experiment, but not both together." (Section 3.1 CLEVR)
- A single jointly trained model instance spanning CLEVR, Sort-of-CLEVR, bAbI, and dynamic physical systems: "Not specified in the paper."

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{24\ \text{tasks}}{5\ \text{models}} = 4.8
}
$$
