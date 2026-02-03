# RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs? (Not specified in the paper.)
Source: RL Grokking Recipe- How Does RL Unlock and Transfer New Algorithms in LLMs-.md

## Core reasons
- The paper’s main contribution is a new controlled benchmark/dataset (DELTA) of synthetic programming problem families to test learnability and transferability.
- The work centers on evaluation protocols and controlled OOD splits to measure generalization, positioning it as benchmark and measurement infrastructure.

## Evidence extracts
- "we introduce DELTA — Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding, a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability—can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)?—and transferability— if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets?" (Abstract)
- "**A controlled dataset (DELTA)**: We design a suite of synthetic programming problem families that isolate reasoning skills, enabling clean tests of learnability (can RL unlock procedures absent in the base model) and generalization (do these procedures transfer systematically to OOD cases)." (Main contributions)

## Classification
Class name: Data, Benchmarks & Measurement
Class code: 4

$$
\boxed{4}
$$
