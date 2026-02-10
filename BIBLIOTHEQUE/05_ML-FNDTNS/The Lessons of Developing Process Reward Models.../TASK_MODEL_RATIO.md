1. **Number of distinct tasks evaluated: 15.**
"Following Yang et al. (2024c), we sampled eight responses (i.e., N=8) from Qwen2.5-Math-7B-Instruct across multiple mathematical benchmarks, including GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), Minerva Math (Lewkowycz et al., 2022), GaoKao 2023 En (Liao et al., 2024), OlympiadBench (He et al., 2024), College Math (Tang et al., 2024), and MMLU STEM (Hendrycks et al., 2021a)." (Section 2.2)
"To validate the effectiveness of our trained PRM Qwen2.5-Math-PRM-7B and Qwen2.5-Math-PRM-72B, we respectively conduct the response-level BoN evaluation and the step-level process errors identification task PROCESSBENCH (Zheng et al., 2024)." (Section 4.2)
"Then we randomly choose correct-answer responses from them and conduct thorough manual annotations." and "we sample 8 responses per query from GSM8K, MATH, OlympiadBench, and Omni-MATH using the policy model Qwen2.5-Math-7B-Instruct." (Section 3.2.1)
"To validate the effectiveness of our PRMs on the BoN with larger N values, we conduct additional Best-of-8 experiments on the policy model Qwen2.5-Math-7b-Instruct across diverse tasks including MATH500 (Lightman et al., 2023), AIME24  $^1$ , AMC23  $^2$ , Minerva Math (Lewkowycz et al., 2022), GaoKao 2023 En (Liao et al., 2024) and OlympiadBench (He et al., 2024)." (Section B.4)
"We evaluate across three Chinese benchmarks including Chinese math benchmarks CMATH (Wei et al., 2023), GaoKao Math Cloze (Zhong et al., 2024), and GaoKao Math QA (Zhong et al., 2024) following Yang et al. (2024c), as shown in Table 15 and Table 16." (Section B.3)

2. **Number of trained model instances required to cover all tasks: 1 model.**
"Qwen2.5-Math-PRM-7B demonstrates superior performance compared to other PRMs of equivalent model scale. Notably, it outperforms maj@8 across all 7 tasks" (Section 4.3), and the same trained PRM family is used for both "response-level BoN evaluation" and "step-level process errors identification task PROCESSBENCH" (Section 4.2), with no task-specific training per benchmark described.

3. **Task–Model Ratio**

$$
\boxed{
\frac{15\ \text{tasks}}{1\ \text{model}} = 15
}
$$
