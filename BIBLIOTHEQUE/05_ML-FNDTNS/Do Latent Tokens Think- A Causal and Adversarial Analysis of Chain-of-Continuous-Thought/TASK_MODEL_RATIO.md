1. **Number of distinct tasks evaluated:** 4

   - "For perturbation experiments, we use two datasets with strong directional tendencies: the AdvBench (Chen et al., 2022) dataset, and the PersonalityEdit (Mao et al., 2024) dataset. For token-swapping experiments, we use the MMLU (Hendrycks et al., 2020) dataset." (Section 4.2, Experiments)
   - "For multiple-choice experiments (*option manipulation*), we use the MMLU (Hendrycks et al., 2020) dataset. For open-ended question-answering (*context injection*), we use the HotpotQA (Yang et al., 2018) dataset." (Section 5.2, Experiments)

2. **Number of trained model instances required to cover all tasks:** 3

   - "Our approach consists of three main components: (i) aligning the model's reasoning behavior via task-specific fine-tuning;" (Section 4.1, Method)
   - "For the steering experiments, each model is trained for 6 epochs on ProntoQA. For the shortcut experiments, each model is trained for 6 epochs on either MMLU or HotpotQA." (Appendix C, Training Setups)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{4\ \text{tasks}}{3\ \text{models}} = 1.33
}
$$
