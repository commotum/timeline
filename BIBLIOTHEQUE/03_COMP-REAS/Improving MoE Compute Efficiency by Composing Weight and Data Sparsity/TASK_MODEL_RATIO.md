1. **Number of distinct tasks evaluated:** 23

   Verbatim evidence:
   - "Table 1 compares our models against Isaac 0.2, a dense 1.7B baseline trained with the same recipe, and InternVL3.5-20B-A4B [29], a similarly sized sparse VLM." (Section 5.4, HERO RUN)
   - "Both MoE configurations consistently outperform the dense baseline across all task categories, with the  $\mathbf{K}_{0.5}^8$  model achieving the best overall results." (Section 5.4, HERO RUN)
   - Table 1 benchmark rows enumerate the evaluated tasks: "Aerial Grounding," "Perceptron Grounding," "RefCOCO," "ChartQA," "DocVQA," "A-OKVQA (val)," "TextVQA," "OCRBench," "Aerial Counting," "CVBench," "PixMoCount," "CountBench," "VSR (Zero-Shot)," "VQA v2," "RealWorldQA," "SEED-Bench," "M3Exam (English)," "NLVR2," "BLINK," "MathVista," "MME," "AI2D," "ERQA." (Section 5.4, Table 1)

2. **Number of trained model instances required to cover all tasks:** 1 model

   Verbatim evidence:
   - "Both MoE configurations consistently outperform the dense baseline across all task categories, with the $\mathbf{K}_{0.5}^8$ model achieving the best overall results." (Section 5.4, HERO RUN)
   - "| Task     | Benchmark             | 1.7B $\mathbf{K}_{1.0}^4$ | 1.7B $\mathbf{K}_{0.5}^{8}$ | Isaac 0.2 | InternVL3.5-20B-A4B |" (Section 5.4, Table 1)
   - Task-specific heads/decoders per benchmark: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{23\ \text{tasks}}{1\ \text{model}} = 23
}
$$
