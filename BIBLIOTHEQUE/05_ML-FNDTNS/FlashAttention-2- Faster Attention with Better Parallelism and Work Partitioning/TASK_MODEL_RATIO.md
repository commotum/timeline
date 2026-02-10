1. **Number of distinct tasks evaluated:** 2

   "• Benchmarking attention. We measure the runtime of FlashAttention-2 across different sequence lengths and compare it to a standard implementation in PyTorch, FlashAttention, and FlashAttention in Triton. We confirm that FlashAttention-2 is 1.7-3.0× faster than FlashAttention, 1.3-2.5× faster than FlashAttention in Triton, and 3-10× faster than a standard attention implementation." (Section 4, `# 4 Empirical Validation`)

   "• End-to-end training speed When used end-to-end to train GPT-style models of size 1.3B and 2.7B on sequence lengths either 2k or 8k, FlashAttention-2 yields up to 1.3× speedup compared to FlashAttention and 2.8× speedup compared to a baseline without FlashAttention. FlashAttention-2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization) per A100 GPU." (Section 4, `# 4 Empirical Validation`)

2. **Number of trained model instances required to cover all tasks:** 1 model

   "We measure the runtime of different attention methods on an A100 80GB SXM4 GPU for different settings (without / with causal mask, head dimension 64 or 128)." (Section 4.1, `## 4.1 Benchmarking Attention`)

   "We measure the training throughput of GPT-style models with either 1.3B or 2.7B parameters, on 8×A100 80GB SXM4." (Section 4.2, `#### 4.2 End-to-end Performance`)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{1\ \text{model}} = 2
}
$$
