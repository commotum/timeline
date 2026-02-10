1. **Number of distinct tasks evaluated:** 1

   "To comprehensively evaluate our method, we establish a new multi-scene, multi-shot story video generation benchmark, termed ST-Bench." (Section 4.2 ST-Bench)

2. **Number of trained model instances required to cover all tasks:** 1

   "Our framework is built upon the state-of-the-art open-source video generation model Wan2.2-I2V-A14B [42] with 14B active parameters. We finetune it using a rank-128 LoRA applied to all linear layers in the DiT blocks, adding  $\sim$ 0.7B active parameters. The M2V model is trained on a curated dataset of 400K five-second single-shot videos, where each clip is matched with 1–5 semantically coherent videos." (Section 4.1 Implementation Details)

   "Another application is to personalize the initialization of the memory state  $m_0$ . For instance, users can provide character or background reference images as the initial memory, enabling customized multi-shot video generation." (Section 3.5 Extension to MI2V and MR2V)

3. **Task–Model Ratio**

$$
\boxed{
\frac{1\ \text{tasks}}{1\ \text{models}} = 1
}
$$
