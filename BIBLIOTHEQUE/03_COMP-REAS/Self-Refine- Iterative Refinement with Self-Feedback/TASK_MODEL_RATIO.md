1. **Number of distinct tasks evaluated:** 7.  
   Quote (Section 3 Evaluation): "We evaluate Self-Refine on 7 diverse tasks: Dialogue Response Generation (Appendix M; Mehri and Eskenazi, 2020), Code Optimization (Appendix N; Madaan et al., 2023), Code Readability Improvement (Appendix L; Puri et al., 2021), Math Reasoning (Appendix O; Cobbe et al., 2021), Sentiment Reversal (Appendix P; Zhang et al., 2015), and we introduce two new tasks: Acronym Generation (Appendix Q) and Constrained Generation (a harder version of Lin et al. (2020) with 20-30 keyword constraints instead of 3-5; Appendix R)"

2. **Number of trained model instances required to cover all tasks:** 1 model.  
   Quote (Abstract): "Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider."  
   Quote (Section 2 Iterative Refinement with SELF-REFINE): "The key idea is that Self-Refine uses the same underlying LLM to generate, get feedback, and refine its outputs given its own feedback."

3. **Task–Model Ratio = (1) / (2):** 7 / 1 = 7.

$$
\boxed{
\frac{7\ \text{tasks}}{1\ \text{model}} = 7
}
$$
