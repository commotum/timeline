1. **Number of distinct tasks evaluated:** 1

   "To diagnose the failures of current models and support research, we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems." (Abstract)

   "We investigate two methods to solve problems in GSM8K: finetuning and verification." (Section 4, Methods)

2. **Number of trained model instances required to cover all tasks:** 2

   "As shown in Figure 4, we train the verifier as follows:" (Section 4.2, Verification)

   "1. Finetune a model (the \"generator\") for 2 epochs on the training set." (Section 4.2, Verification)

   "3. Train a verifier for a single epoch on this dataset." (Section 4.2, Verification)

   "We train separate generator and verifier models to limit the generator's training and prevent overfitting, but in principle, it should be possible to combine these models." (Section 4.2, Verification)

   "At test time, we sample 100 completions to each test problem, rank them with the verifier, and then return the one with the highest verifier score." (Section 4.2, Verification)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{1\ \text{task}}{2\ \text{models}} = 0.5
}
$$
