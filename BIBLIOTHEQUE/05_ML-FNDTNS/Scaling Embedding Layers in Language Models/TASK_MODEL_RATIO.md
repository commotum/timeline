1. **Number of distinct tasks evaluated:** 9

> "We focus on pre-training decoder-only language models with causal language modeling [Radford et al., 2019]." (Section 2 Preliminaries)

> "We report zero-shot accuracy on six standard downstream benchmarks: MMLUvar, Hellaswag, ARC-Challenge, ARC-Easy, CommonsenseQA (CSQA), and PIQA." (Section 4.2 Scaling Up the Training Corpus)

> "Table 4: Performance comparison of Qwen3-4B variants on AIME 2024 and LiveCodeBench v4/v5 with decoding latency." (Section E.3 Apply Scone in Post-training)

2. **Number of trained model instances required to cover all tasks:** 2

> "In this section, we evaluate SCONE in pre-training settings." (Section 4 Experimental Evaluation)

> "For completeness, we also evaluate SCONE in post-training settings by applying it during the SFT stage of recent Qwen3 models [Yang et al., 2025]; these results, presented in Section E.3, show that SCONE remains effective in post-training as well." (Section 4 Experimental Evaluation)

> "We apply SCONE to supervised fine-tuning of Qwen3-4B-base..." (Section E.3 Apply Scone in Post-training)

3. **Task–Model Ratio**

$$
\boxed{
\frac{9\ \text{tasks}}{2\ \text{models}} = 4.5
}
$$
