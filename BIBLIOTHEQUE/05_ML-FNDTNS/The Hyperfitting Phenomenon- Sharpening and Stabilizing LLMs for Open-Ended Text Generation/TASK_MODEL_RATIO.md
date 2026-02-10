1. **Number of distinct tasks evaluated:** 10

- "## 4 OPEN-ENDED TEXT GENERATION" (Section 4)
- "#### 7.1 IMAGE GENERATION" (Section 7.1)
- "We further explore the effect of hyperfitting on downstream tasks using the MMLU (Hendrycks et al., 2021) benchmark and GLUE benchmark (Wang et al., 2018)." (Section B.4)
- "The GLUE dataset tasks the model across a range of different tasks." (Section B.4.2)
- "Table 8: Comparison of the original and hyperfitted Llama and Deepseek models across various GLUE tasks" with task rows "cola", "mnli", "mrpc", "qnli", "qqp", "rte", and "sst2" (Table 8)

2. **Number of trained model instances required to cover all tasks:** 2

- "For both these tests we do not apply any further fine-tuning, and instead use the model's hyperfitted on the Fiction dataset as described in Section 3." (Section B.4)
- "Specifically, we fine-tune one instance for each of the following models: Tiny Llama 1.1b, DeepSeek 7b (Bi et al., 2024), Llama 3.1 8b & 70B (Dubey et al., 2024), and ImageGPT-Large (Chen et al., 2020) for image generation." (Section 3)
- "To investigate the hyperfitting phenomenon for an additional modality, we hyperfit ImageGPT-Large (774M parameters) (Chen et al., 2020) on 2,000 randomly selected images from CIFAR-10." (Section 7.1)

3. **Task–Model Ratio**

$$
\boxed{
\frac{10\ \text{tasks}}{2\ \text{models}} = 5
}
$$
