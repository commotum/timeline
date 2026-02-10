1. Number of distinct tasks evaluated: 2.
- "We design a simple experiment where the model needs to learn n vector tuples  $\{(q_i, k_i, v_i)\}_{i=1}^n$  such that the attention head can retrieve  $v_i$  with  $q_i$  from any randomly sampled subset of key-value pairs" (§4 Controlled Experiment).
- "As we hypothesize that the dimension inefficiency only occurs for attention heads that model long dependency, we choose a task that involves long dependence modeling, the long-context question-answering task." (§5.1 Experimental Setup).

2. Number of trained model instances required to cover all tasks: 2 models.
- "We train models in two setups, one with RoPE applied on K and the other without (details in  $\S A$ )." (§4 Controlled Experiment). 
- "We then inspect three 7B/8B large language models (LLM), Llama-3.1-8B-Instruct (Dubey et al., 2024), QWen-2.5-7B-Instruct (Team, 2024), and OLMo-2-7B-Instruct (OLMo et al., 2024)." (§5 Inspecting Real-world Models).
- Whether a single jointly trained model instance is used to cover both tasks: Not specified in the paper.

3. Task–Model Ratio

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
