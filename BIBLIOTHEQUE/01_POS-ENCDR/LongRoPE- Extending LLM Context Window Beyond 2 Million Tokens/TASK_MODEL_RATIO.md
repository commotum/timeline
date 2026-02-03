1. Number of distinct tasks evaluated: 3. "We apply LongRoPE on LLaMA2-7B and Mistral-7B, and evaluate the performance on three aspects: (1) perplexity of extended-context LLMs on long documents; (2) Passkey retrieval task that measures a model's ability to retrieve a simple passkey from a sea of irrelevant text; and (3) Standard LLM benchmarks within a short 4096 context window size." (Section 4.1. Setup)
2. Number of trained model instances required to cover all tasks: 1 model. "As shown in Table 6, LongRoPE successfully extends LLaMA2-7B and Mistral-7B's context window to 2048k, while also achieving perplexity comparable or superior to baselines within shorter lengths of 8k-128k." (Section 4.2. Main Results) "our LongRoPE-LLaMA2-2048k (ft=256k) manage to maintain a high retrieval accuracy (≥90%) from 4k to 2048k." (Section 4.2. Passkey retrieval) "We evaluate LongRoPE-2048k models on the original context window using Hugging Face Open LLM Leader-board (Face, 2024) in zero-shot and few-shot settings." (Standard benchmarks within original context window)
3. Task–Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$
