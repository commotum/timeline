1. **Number of distinct tasks evaluated:** 21

"To assess downstream performance, we evaluate both models on benchmarks spanning three core capability domains:" (Section 6.2, p.12)

"- General Tasks: MMLU [Hendrycks et al., 2021], MMLU-Pro [Wang et al., 2024], C-Eval [Huang et al., 2023], and CMMLU [Li et al., 2023].
- Reasoning Tasks: BBH [Suzgun et al., 2023], GPQA [M-A-P Team, ByteDance., 2025], DROP [Dua et al., 2019] and GSM8K [Cobbe et al., 2021].
- Coding Tasks: HumanEval+ [Liu et al., 2024], MultiPL-E [Cassano et al., 2022], and BigCodeBench [Zhuo et al., 2025]." (Section 6.2, p.12)

"The evaluation of the chat model covers several core capabilities: agentic tool use tasks, agentic coding tasks, general domain tasks and mathematical reasoning tasks. The benchmarks used for assessment include:" (Section 6.3, p.13)

"- Agentic Tool Use Tasks:  $\tau^2$  Bench [Barres et al., 2025], Vita Bench [He et al., 2025].
- Agentic Coding Tasks: SWE-Bench [Jimenez et al., 2023], TerminalBench [Merrill et al., 2026], SWE-Bench Multiligual [Yang et al., 2025], and PRDBench [Fu et al., 2025].
- General Domain Tasks: GPQA-Diamond [Rein et al., 2024], MMLU [Hendrycks et al., 2021], MMLU-Pro [Wang et al., 2024], C-Eval [Huang et al., 2023], and CMMLU [Li et al., 2023].
- Mathematical Reasoning Tasks: MATH500 [Lightman et al., 2023], AIME24 [MAA, 2024], AIME25 [MAA, 2025]." (Section 6.3, p.13)

Counted as unique benchmarks across Sections 6.2 and 6.3: 11 + 14 - 4 overlaps (MMLU, MMLU-Pro, C-Eval, CMMLU) = 21.

2. **Number of trained model instances required to cover all tasks:** 2

"### **6.2** Base Model Evaluation" (Section 6.2, p.12)

"# 6.3 Chat Model Evaluation" (Section 6.3, p.13)

"LongCat-Flash-Lite undergoes a complete pipeline of pre-training, mid-training, and supervised finetuning, and demonstrates highly competitive performance for its scale." (Section 6, p.12)

Whether one single checkpoint was used to run both base-model and chat-model benchmark suites is Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{21\ \text{tasks}}{2\ \text{models}} = 10.5
}
$$
