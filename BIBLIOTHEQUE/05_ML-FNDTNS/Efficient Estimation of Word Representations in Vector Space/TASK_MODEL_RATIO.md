1. **Number of distinct tasks evaluated:** 2

   - Task 1 (word relationship / similarity evaluation): "The quality of these representations is measured in a word similarity task" (Abstract).
   - Task 1 detail: "To measure quality of the word vectors, we define a comprehensive test set that contains five types of semantic questions, and nine types of syntactic questions." (Section 4.1, Task Description).
   - Task 2 (sentence completion): "The Microsoft Sentence Completion Challenge has been recently introduced as a task for advancing language modeling and other NLP techniques [32]." (Section 4.5, Microsoft Research Sentence Completion Challenge).

2. **Number of trained model instances required to cover all tasks:** 2

   - For the word relationship/similarity evaluations, trained word-vector models are used: "We have used a Google News corpus for training the word vectors." (Section 4.2, Maximization of Accuracy).
   - For sentence completion, a separate task-specific training run is described: "We have explored the performance of Skip-gram architecture on this task. First, we train the 640-dimensional model on 50M words provided in [32]." (Section 4.5, Microsoft Research Sentence Completion Challenge).

3. **Task-Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
