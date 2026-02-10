1. **Number of distinct tasks evaluated:** 2

> "To validate the accuracy of our method in process reward modeling, we conduct evaluations on two challenging benchmarks *ProcessBench* (Zheng et al., 2024) and *PRM-Bench* (Song et al., 2025)." (Section 4.1, **Tasks and Benchmarks**)

> "Furthermore, we validate the effectiveness of our model by employing it to guide two distinct test-time scaling paradigm: Best-of-N and Guide Search." (Section 4.1, **Tasks and Benchmarks**)

2. **Number of trained model instances required to cover all tasks:** 2

> "Qwen2.5-Math-7B-Instruct is fine-tuned for one epoch with batch size 128 and learning rates of 5e-6 (SFT) and 5e-7 (DPO)." (Section 4.1, Implementation details)

> "Consistent with previous work (Zhang et al., 2025b), we used Qwen2.5-7B-Instruct to generate eight candidate steps with temperature T=1.0." (Section 4.1, **Tasks and Benchmarks**)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
