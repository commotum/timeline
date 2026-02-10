1. **Number of distinct tasks evaluated:** 10

   - (Section 3, "Results") "We measure performance on the SuperGLUE benchmark (Wang et al., 2019a), a collection of eight challenging English language understanding tasks."
   - (Section 5, "Resilience to Domain Shift") "We investigate zero-shot domain transfer on two tasks: question answering (QA) and paraphrase detection."

2. **Number of trained model instances required to cover all tasks:** 10

   - (Section 3, "Results") "Each of our prompts train on a single Super-GLUE task; there was no multi-task setup or mixing of training data across tasks."
   - (Section 5, "Resilience to Domain Shift") "For our experiments, we train on SQuAD (Rajpurkar et al., 2016) and evaluate on each of the out-of-domain datasets."
   - (Section 5, "Resilience to Domain Shift") "As before, we train on the \"in-domain\" task, select checkpoints using in-domain validation, and evaluate zero-shot on the \"out-of-domain\" task."

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{10\ \text{tasks}}{10\ \text{models}} = 1
}
$$
