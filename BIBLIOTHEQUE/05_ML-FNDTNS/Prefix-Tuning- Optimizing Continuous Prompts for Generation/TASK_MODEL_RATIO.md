1. **Number of distinct tasks evaluated:** 2

> "Consider a conditional generation task where the input is a context x and the output y is a sequence of tokens. We focus on two tasks, shown in Figure 2 (right): In table-to-text, x corresponds to a linearized data table and y is a textual description; in summarization, x is an article and y is a short summary." (Section 3, Problem Statement)

2. **Number of trained model instances required to cover all tasks:** 2 models

> "In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix)." (Abstract)

> "We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization." (Abstract)

> "For table-to-text, we use GPT- $2_{\rm MEDIUM}$  and GPT- $2_{\rm LARGE}$ ; the source tables are linearized.<sup>7</sup> For summarization, we use BART<sub>LARGE</sub>, and the source articles are truncated to 512 BPE tokens." (Section 5.3, Architectures and Hyperparameters)

3. **Task–Model Ratio**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
