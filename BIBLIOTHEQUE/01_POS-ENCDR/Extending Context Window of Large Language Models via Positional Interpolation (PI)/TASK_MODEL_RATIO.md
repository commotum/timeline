- “We demonstrate its performance in a number of tasks including language modeling, passkey retrieval, and long document summarization.” (Section 3 EXPERIMENTS)
- “We evaluate the models extended by Position Interpolation on several standard benchmark tasks within the original context window size of 2048. The evaluation results are listed in Table 5.” (Section 3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE)
- “BoolQ | PIQA | Race-M | Race-H | WinoGrande” (Table 5)
- “We fine-tune all model variants using the next token prediction objective.” (Section 3.1 SETUP)
- “We fine-tune the LLaMA models extended with Position Interpolation with a context window of 16384.” (Section 3.5 Long Document Summarization)

Number of distinct tasks evaluated: 8 (language modeling, passkey retrieval, long document summarization, BoolQ, PIQA, Race-M, Race-H, WinoGrande).

Number of trained model instances required to cover all tasks: 2.

$$
\boxed{
\frac{8\ \text{tasks}}{2\ \text{models}} = 4
}
$$
