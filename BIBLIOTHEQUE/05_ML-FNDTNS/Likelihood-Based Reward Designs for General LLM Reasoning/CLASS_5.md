# Likelihood-Based Reward Designs for General LLM Reasoning (2026)
Source: Likelihood-Based Reward Designs for General LLM Reasoning.md

## Core reasons
- The paper's main contribution is a training-objective design for LLM post-training: it studies and advocates likelihood/log-likelihood rewards for RL fine-tuning across reasoning tasks.
- It is primarily an ML training/optimization methodology study (reward formulation, RL objective behavior, perplexity/success trade-offs), not a positional encoding change, dimensional transformer adaptation, or benchmark/resource paper.

## Evidence extracts
- "We find that using the *log-probability* of the reference answer as the reward for chain-of-thought (CoT) learning is the only option that performs well in all setups." (Abstract)
- "For instance, we can set a reward similar to the log-loss used during pretraining," (Section 2 Method)
- "Our work establishes log-probability rewards as a unifying training signal effective in both verifiable and non-verifiable domains, without relying on ground-truth correctness labels." (Section 4 Conclusion)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
