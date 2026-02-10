1. **Number of distinct tasks evaluated:** 4

   "Datasets. We consider two *verifiable* math benchmarks and two *non-verifiable* long-form datasets. (i) MATH (Hendrycks et al., 2021b):We report accuracy on the official test split. The resulting training set contains ~7,000 short-answer problems. (ii) DeepScaleR (Preview) (Luo et al., 2025): we hold out a random 10% for validation to report performance. The training set has ~39,000 short-answer problems. (iii) Alpaca (cleaned) (Taori et al., 2023): we use the standard cleaned variant; 1,000 random examples are used for validation, leaving ~50,000 training samples with predominantly long-form answers. (iv) NuminaProof: starting from NuminaMath-1.5 (Li et al., 2024), we filter for theorem-proof style items. We reserve 1,000 examples for validation, yielding ~50,000 long-form training samples. More detail in Section B." (Section 3.1: Setup: Datasets, Models, and Protocol)

2. **Number of trained model instances required to cover all tasks:** 4

   "For each experiment, we use a synchronous implementation of RLOO running in parallel across 8 processes." (Section B: Experimental details)

   "Each batch contains 8 questions from the dataset with G different CoTs; such a batch corresponds to one step in all our figures." (Section B: Experimental details)

   "Here, we complement Figure 1 with the corresponding learning curves for other model-dataset combinations (Figures 3 to 5) and provide the corresponding Figures 6 to 9 and Table 3 for training with G = 4 (including JEPO, which for efficiency reasons we only ran for G = 4)." (Section C.1: Verifiable Domains)

   "Llama 3B, MATH" / "Llama 3B, DeepScaleR" (Table 1, Section 3.1) and "Llama 3B, NuminaProof" / "Llama 3B, Alpaca" (Table 2, Section 3.3)

3. **Task–Model Ratio = (1) / (2):** 1

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
