1. **Number of distinct tasks evaluated:** 2

- "We experimentally investigate these concepts in two distinct environments: Partially-Observable Graph Search (POGS), a synthetic environment that we design to systematically evaluate planning abilities, and Crafter, a Minecraft-inspired grid-world environment (Hafner, 2022)." (Section 1)
- "To evaluate dynamic planning across different conditions, we select two complementary environments." (Section 4.1)
- "First, **Partially Observable Graph Search (POGS)** is our custom synthetic environment designed to isolate planning under uncertainty." (Section 4.1)
- "Second, **Crafter** (Hafner, 2022) is a complex 2D grid-world, long-horizon benchmark inspired by Minecraft." (Section 4.1)

2. **Number of trained model instances required to cover all tasks:** 2

- "To understand baseline capabilities and the raw effect of planning frequency, we perform zero-shot evaluations using Llama-3.3-70B-Instruct (Grattafiori et al., 2024) on POGS and Crafter (100 seeds each)." (Section 4.3)
- "**SFT Priming:** The Llama-3.1-8B model was fine-tuned on this data, aligning the SFT process with the target RL configuration." (Section 4.4)
- "We then used Proximal Policy Optimization (PPO) (Schulman et al., 2017) to fine-tune Llama-3.1-8B-Instruct agents in Crafter, optimizing task rewards possibly adjusted for planning costs (Sec. 3.3)." (Section 4.5)

3. **Task–Model Ratio**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
