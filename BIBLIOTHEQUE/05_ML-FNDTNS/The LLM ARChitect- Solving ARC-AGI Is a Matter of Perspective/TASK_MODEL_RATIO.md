1. **Number of distinct tasks evaluated:** 500
   - "The original ARC-AGI dataset consists of 900 reasoning tasks, divided into 400 training tasks, 400 public evaluation tasks, and 100 private evaluation tasks." (Section 3.1 Datasets)
   - "achieving 53.5 (56.5¹) points on the private evaluation set during the Kaggle ARC Prize 2024 Contest. Additionally, we are able to solve 72.5 out of 100 randomly split-off tasks from the public evaluation set" (Abstract)
   - "full public evaluation set is used for Llama-rearc" (Table 5 caption, Section 3.5 Solution Inference)

2. **Number of trained model instances required to cover all tasks:** 1
   - "We retrain an LLM on public ARC-AGI data, which is then finetuned an additional time on the hidden test cases. Subsequently, this model predicts several solution candidates" (Figure 1 caption, Section 2 Pipeline Overview)
   - "given the generated list of candidates, we use the aggregated logsoftmax scores assigned by the fine-tuned model to select two of them for submission" (Section 2, Candidate Selection)
   - "We also tried an additional separate fine-tuning for each task, which increased the score slightly. However, it did not do enough to be considered runtime efficient in our approach and was discarded." (Section 3.4 Training the models)

3. **Task–Model Ratio = (1) / (2):** 500

$$
\boxed{
\frac{500\ \text{tasks}}{1\ \text{model}} = 500
}
$$
