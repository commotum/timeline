1. **Number of distinct tasks evaluated:** Not specified in the paper.

   - "our full evaluation suite of tasks" (Section 3.4, Human Baselines)
   - "we use a subset of 50 programmatic tasks" (Section 3.1.1, Held-Out Environments)
   - "On a representative subset of tasks, we found human performance to be roughly 19% for MineDojo (16 tasks) and 32% for ASKA (25 tasks)." (Section 4.2.2, Performance in Held-Out Environments)

2. **Number of trained model instances required to cover all tasks:** 2 models.

   - "At its core, the SIMA 2 agent architecture is a Gemini Flash-Lite model" (Section 3.3, Data, Agent & Training)
   - "we explore composing SIMA 2 with a separate Gemini Pro model" (Section 4.4, Gemini Instructing SIMA 2)
   - "the combined Gemini Pro + SIMA 2 agent" (Appendix B, Additional Results Combining Gemini Pro & SIMA 2)

3. **Task–Model Ratio**

$$
\boxed{
\frac{N\ \text{tasks}}{2\ \text{models}} = \frac{N}{2}
}
$$
