1. **Number of distinct tasks evaluated:** 27

> "To demonstrate adapter's effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark." (Abstract)

> "Finally, we confirm that adapters work on tasks other than classification by running on SQuAD v1.1 (Rajpurkar et al., 2018)." (Section 3.5. SQuAD Extractive Question Answering)

2. **Number of trained model instances required to cover all tasks:** 27

> "Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones." (Abstract)

> "During adapter tuning, the green layers are trained on the downstream data, this includes the adapter, the layer normalization parameters, and the final classification layer (not shown in the figure)." (Figure 2, Section 2.1. Instantiation for Transformer Networks)

3. **Task–Model Ratio**

$$
\boxed{
\frac{27\ \text{tasks}}{27\ \text{models}} = 1
}
$$
