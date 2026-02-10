1. **Number of distinct tasks evaluated: 4**

   "Following TinyLlama, we evaluate the commonsense reasoning ability of the NoPE model and report acc_norm in Table 1." (Section 4.1, NoPE pre-trained model)

   "We conduct length generalization experiments on long sequence language modeling, synthetic tasks (passkey retrieval), and real-world long context tasks (LongBench)." (Section 4, Experiment)

2. **Number of trained model instances required to cover all tasks: 1**

   "For a fair comparison with RoPE, we train a NoPE model with 1.1B parameters from the TinyLlama (Zhang et al., 2024b) code base." (Section 4.1, NoPE pre-trained model)

   "To evaluate the long sequence language modeling performances, we test our NoPE-based methods and RoPE-based baselines on PG19 (Rae et al., 2020) and proof-pile (Azerbayev et al., 2022) datasets." (Section 4.2, Long Sequence Language Modeling)

   "We evaluate the performance of passkey retrieval across various context lengths." (Section 4.3, Synthetic Long Context Tasks)

   "LongBench (Bai et al., 2023) is a comprehensive assessment of the long context understanding capabilities of large language models. We test all models using beam search decoding with beam size 5." (Section 4.4, Real-World Long Context Tasks)

3. **Task–Model Ratio**

$$
\boxed{
\frac{4\ \text{tasks}}{1\ \text{model}} = 4
}
$$
