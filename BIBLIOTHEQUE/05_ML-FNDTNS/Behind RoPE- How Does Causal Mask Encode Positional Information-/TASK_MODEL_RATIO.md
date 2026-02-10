1. **Number of distinct tasks evaluated:** 1 task.

"We conducted an empirical study to examine whether similar attention patterns arise when training a Transformer decoder without positional encoding." (Section 4.2, *Analysis of a Trained Model Without Positional Encoding*)

"We analyze whether the same phenomenon is observed in modern LLMs trained with RoPE, including Llama-3.1 8B Grattafiori et al. (2024), Phi-4 (Abdin et al., 2024), and Qwen3-8B (Yang et al., 2025a)." (Section 5.2, *Analysis of LLMs*)

2. **Number of trained model instances required to cover all tasks:** 1 model.

"we trained a model based on the Llama-3 architecture (Grattafiori et al., 2024) having 1.5B parameters" (Section 4.2, *Analysis of a Trained Model Without Positional Encoding*)

"We further examine the behavior of positional information from the causal mask through simulation of a Transformer without parameters and without positional encodings" (Section 4.1, *Transformer Simulation without Parameters*)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
