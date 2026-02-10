1. **Number of distinct tasks evaluated:** Not specified in the paper.
   - "Across the 7 environments for which we have evaluations, we have a total of 1,485 unique tasks, spanning a range of 9 skill categories, from movement (\"go ahead\", \"look up\", \"jump\") to navigation (\"go to the HUB terminal\", \"go to your ship\"), resource gathering (\"collect carbon\", \"get raspberries\"), object management (\"use the analysis visor\", \"cut the potato\"), and more." (Section 3.4. Evaluation methods)
   - "To provide an additional baseline comparison, we evaluated our agents against expert human performance on an additional set of tasks from No Man's Sky, which were chosen to test a focused set of skills in a diverse range of settings." (Section 4.3. Human comparison)

2. **Number of trained model instances required to cover all tasks:** 1 model.
   - "SIMA: Our main SIMA agent, which is trained across all environments except for Hydroneer and Wobbly Life, which we use for qualitative zero-shot evaluation." (Section 4.2. Evaluating environment generalization & ablations)
   - "The SIMA agent maps visual observations and language instructions to keyboard-and-mouse actions (Figure 4)." (Section 3.3. Agent)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{N\ \text{tasks}}{1\ \text{model}} = N
}
$$
