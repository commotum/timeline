1. **Number of distinct tasks evaluated:** 2

   "We study this question on ARC by training neural models for *induction* (inferring latent functions) and *transduction* (directly predicting the test output for a given test input)." (Section: **ABSTRACT**)

   "We use the same neural network architecture and dataset to perform both tasks, allowing a controlled comparison between these paradigms." (Section: **7 RELATED WORK**)

2. **Number of trained model instances required to cover all tasks:** 2 models

   "We then meta-learn by further fine-tuning Llama3.1-8B-instruct for induction or transduction using a synthetically-generated corpus of problems, described next." (Section: **2 NEURAL MODELS FOR INDUCTION AND TRANSDUCTION**)

   "Therefore we ensemble by attempting induction first, then transduction if none of the candidate hypotheses explained the examples:" (Section: **2 NEURAL MODELS FOR INDUCTION AND TRANSDUCTION**)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
