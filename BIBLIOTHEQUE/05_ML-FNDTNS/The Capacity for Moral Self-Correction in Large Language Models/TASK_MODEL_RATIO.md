1. **Number of distinct tasks evaluated:** 3
- "We test our hypothesis with three experiments (§3) that measure the propensity for large language models to use negative stereotypes or to discriminate based on protected demographic attributes." (Section 1, Introduction)
- "We use the Bias Benchmark for QA (BBQ) benchmark [40] to measure stereotype bias across 9 social dimensions (§3.2.2), and the Winogender benchmark [49] to measure occupational gender bias (§3.2.3)." (Section 1, Introduction)
- "We also develop a new benchmark that tests for racial discrimination in language models, derived from a dataset that has been used to study counterfactual fairness [30] (§3.2.4)." (Section 1, Introduction)

2. **Number of trained model instances required to cover all tasks:** 1
- "We study decoder-only transformer models fine-tuned with Reinforcement Learning from Human Feedback (RLHF) [13, 57] to function as helpful dialogue models." (Section 3.1, Models)
- "For each benchmark, we use three simple prompt based interventions that build upon one another." (Section 1, Introduction)
- "The Q+IF and Q+IF+CoT conditions are identical to the ones we use in the BBQ example discussed in §3.2.2." (Section 3.2.3, Winogender)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$
