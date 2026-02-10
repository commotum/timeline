1. **Number of distinct tasks evaluated:** 3

- "We then demonstrate that both kinds of filtering are effective across all three kinds of benchmarks, and that they get *more* effective with scale." (§4. Token-level data filtering works and scales)
- "**Text perplexity** As a proxy for capability, we evaluate small models on their cross-entropy loss on relevant text..." (§3.3. Evaluation)
- "Multiple choice For instruction tuned 1.8B models, we also use multiple choice evaluation." (§3.3. Evaluation)
- "**Free-response** We evaluate our chat trained 1.8B models on free-response answers to HealthSearchQA..." (§3.3. Evaluation)

2. **Number of trained model instances required to cover all tasks:** 3

- "**Pretraining** We train compute-optimal Transformers at scales ranging from 61M to 1.8B parameters..." (§3.2. Model training)
- "For multiple choice training, we use a custom instruction tuning mix..." and "Multiple choice For instruction tuned 1.8B models, we also use multiple choice evaluation." (§3.2. Model training; §3.3. Evaluation)
- "For chat training, we used the smol-smoltalk mix..." and "**Free-response** We evaluate our chat trained 1.8B models on free-response answers to HealthSearchQA..." (§3.2. Model training; §3.3. Evaluation)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
