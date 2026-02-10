1. **Number of distinct tasks evaluated:** 427

"The original training and validation sets consist of 400 tasks each." (Section 4.2)

"BIG-Bench Hard (BBH; Srivastava et al., 2023; Suzgun et al., 2023) is a benchmark comprising 27 challenging tasks across 23 task types" (Section 5.1)

2. **Number of trained model instances required to cover all tasks:** 427

"By default, we learn *task-specific* LoRA adapters for each ARC or BBH task at test-time. That is, we obtain K different LoRA adapters, where K is the number of test tasks." (Section 3.3)

"For each task d, we train a separate set of LoRA parameters at test-time" (Section 5.2)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{427\ \text{tasks}}{427\ \text{models}} = 1
}
$$
