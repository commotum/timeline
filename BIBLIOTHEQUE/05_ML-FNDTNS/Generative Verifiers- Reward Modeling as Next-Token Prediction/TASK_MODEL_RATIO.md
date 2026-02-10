1. **Number of distinct tasks evaluated:** 3

   - "Tasks and Data Generation. We focus on the following reasoning tasks (more details in Ap-166 pendix C):" (Section: Experiments, Tasks and Data Generation)
   - "- Last Letter Concatenation [Wei et al., 2022]: Given a list of words, the task is to concatenate the last letters of each word (for instance, \"Noah Paul Elisha Rebecca\" → \"hlaa\")." (Section: Experiments, Tasks and Data Generation)
   - "- Word Sorting [Suzgun et al., 2022] Given a list of words, sort them in alphabetical order." (Section: Experiments, Tasks and Data Generation)
   - "- GSM8K [Cobbe et al., 2021] is a widely-used dataset to evaluate grade-school math reasoning capabilities of LLMs." (Section: Experiments, Tasks and Data Generation)
   - "GenRM, which directly predicts Yes/No token for verification, can match or outperform the discriminative RM and other approaches on all the three tasks, as shown in Figure 5." (Section: 4.1 Comparing GenRM with Prior Verification Approaches)

2. **Number of trained model instances required to cover all tasks:** 3

   - "We train verifiers on examples of lengths  $\\{2,3,4\\}$  (here the length refers to how many words are in the input list), and evaluate the verifier performance on length 6." (Section: C Training Data Generation for Verifiers)
   - "Word Sorting: We train verifiers on a dataset comprised of  $\\{2,3,4\\}$  words in each example, and evaluate the performance on length 5." (Section: C Training Data Generation for Verifiers)
   - "**Grade School Math**: We follow the original train/test split and use 1.3K problems for test, 128 problems for validation, and about 7.2K problems for training." (Section: C Training Data Generation for Verifiers)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
