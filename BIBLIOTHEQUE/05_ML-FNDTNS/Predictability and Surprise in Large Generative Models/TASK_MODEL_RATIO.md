1. **Number of distinct tasks evaluated:** 6
   - "Here, we illustrate three examples of abrupt capability scaling for arithmetic [11], language understanding, [34, 62], and programming [4]" (Section 2.2).
   - "we consider a problem domain ... recidivism prediction" and "ask language models [3] instead of people to predict recidivism" (Section 2.3).
   - "the toxicity ... of text generated from language models [3] increases smoothly and significantly with model size" (Section 2.4).
   - "Movielens 1M movie recommendation system task [33]" (Appendix A.3).

2. **Number of trained model instances required to cover all tasks:** 3
   - "Fig. 2 ... based on three different models: GPT-3 (blue), Gopher (orange), and a Google language model (green)." Also, "GPT-3 displays a similar phenomenon" on language understanding, so one GPT-3 instance can cover arithmetic + language understanding, and one Google model covers programming (Fig. 2, Section 2.2).
   - Recidivism, recommendation, and toxicity are all evaluated with "language models [3]" (Sections 2.3, 2.4, Appendix A.3, Appendix A.6); model-size variants are not counted as separate instances.

3. **Task–Model Ratio = (1) / (2):** 6 / 3 = 2

$$
\boxed{
\frac{6\ \text{tasks}}{3\ \text{models}} = 2
}
$$
