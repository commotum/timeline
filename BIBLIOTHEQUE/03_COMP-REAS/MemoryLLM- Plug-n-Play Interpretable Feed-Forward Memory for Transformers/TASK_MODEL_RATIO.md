1. **Number of distinct tasks evaluated:** 9

> "Second, we ask: Does the influence of FFNs, viewed as token-indexed key-value memories, remain consistent across different tasks? To investigate, we consider two broad categories of tasks: (a) tasks which heavily rely on recall or retrieval of known information — things that are explicitly stored in FFN memory from the training data (e.g., wikitext-2, LAMBDA, SiQA, ARC-Easy); (b) tasks that require logical, causal, or inferential thinking — where answer isn't directly stored in FFN memory and it must be derived (e.g., HellaSwag, Winogrande, BoolQ, PIQA)." (Section 3.2)

> "| C4<br>Wikitext-2              | $19.730 \\ 25.491$ | 20.933 $27.258$ | 22.079 $29.976$ | $23.190 \\ 32.220$                 |" (Section 4.1, Table 2)

2. **Number of trained model instances required to cover all tasks:** 1

> "Table 1 presents the results of this study of FFNs in a MemoryLLM-1B checkpoint" (Section 3.2)

> "Table 8 Performance comparison of MemoryLLM-1B with uniform low-rank SVD compression of ToLs across 24 layers." (Appendix D.2)

Task-specific heads/decoders/extensions across tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{9\ \text{tasks}}{1\ \text{model}} = 9
}
$$
