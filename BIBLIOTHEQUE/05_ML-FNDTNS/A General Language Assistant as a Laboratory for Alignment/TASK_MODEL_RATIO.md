1. **Number of distinct tasks evaluated:** 15.

   Verbatim evidence: "the authors wrote about fifty comparison evaluations for each category of helpfulness, honesty, harmlessness (HHH), and an 'other' label" (Section 2.2.1); "we include evaluations on TruthfulQA MC1" (Section 2.2.1); "We measured the effect of prompting and context distillation on the toxicity of text generated from language models of increasing size." (Section 2.2.2); "we evaluated relative model performance via a number of head-to-head tests between pairs of models." (Section 2.2.3); "the Codex HumanEval [CTJ<sup>+</sup>21] and the QuixBugs challenge reformulated as a function synthesis task" (Section 2.2.4); and "Binary: Code Correctness, Commonsense (ethics), Justice (ethics), Deontology (ethics), Virtue (ethics), Lambada" plus "Ranked: Learn to Summarize, Utility (ethics), HellaSwag" (Section 3).

2. **Number of trained model instances required to cover all tasks:** 11.

   Verbatim evidence: "In this section we will study a variety of zero-shot evaluations for alignment with and without prompting." (Section 2); "These evaluations were run using our code-finetuned models" (Section 2.2.4); "In this section, all evaluations involve finetuning on a training set and evaluating on a test set" (Section 3.2) across the 9 listed Section 3 tasks; "Our preference models consist of a value head that predicts a single scalar 'score' r" (Section 3.1); and "including with additional heads that can make real-valued predictions at all token positions" (Section 1.2, Models). A single jointly trained model instance covering all tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{15\ \text{tasks}}{11\ \text{models}} = 1.36
}
$$
