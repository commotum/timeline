1. **Number of distinct tasks evaluated:** 5

   "Sequence Copying" (Section 4.1), "Sequence Reversal" (Section 4.1), "Bigram flipping" (Section 4.1), "Subj-Verb-Obj to Subj-Obj-Verb" (Section 4.2), and "Genderless to gendered grammar" (Section 4.2). Section 4.2 states: "We present two simple ITG-based datasets with interesting linguistic properties and their underlying grammars."

2. **Number of trained model instances required to cover all tasks:** 5 models

   Evidence of task-specific training/evaluation: "For each task, test data is generated through the same procedure as training data" (Section 4.3); "For each task, we use as benchmarks the Deep LSTMs described in [1], with 1, 2, 4, and 8 layers." (Section 4.4); and "we present in Figure 2a the coarse- and fine-grained accuracies, for each task, of the best model of each architecture described in this paper" (Section 5). A single jointly trained multi-task model is Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
