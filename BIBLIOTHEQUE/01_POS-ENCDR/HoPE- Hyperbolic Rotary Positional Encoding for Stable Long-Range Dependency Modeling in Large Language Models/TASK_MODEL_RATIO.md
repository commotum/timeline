1. Number of distinct tasks evaluated: 9. Evidence: "we use the Pile dataset (Gao et al., 2020)as the pre-training corpus and evaluate the log perplexity of pre-trained language models in the test sets of PG19 (Rae et al., 2019) and arXiv." (Section 4.2 Perplexity Experiment (PPL)) "It is a long context benchmark that consists of seven distinct datasets covering different tasks" (Section 8.3.2 Fine-Tuning Experiment)
2. Number of trained model instances required to cover all tasks: 8. Evidence: "we use the Pile dataset (Gao et al., 2020)as the pre-training corpus and evaluate the log perplexity of pre-trained language models in the test sets of PG19 (Rae et al., 2019) and arXiv." (Section 4.2 Perplexity Experiment (PPL)) "**Fine-tuning recipes.** We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) "It is a long context benchmark that consists of seven distinct datasets covering different tasks" (Section 8.3.2 Fine-Tuning Experiment)
3. Task–Model Ratio = 9 / 8 = 1.125.

$$
\boxed{
\frac{9\ \text{tasks}}{8\ \text{models}} = 1.125
}
$$
