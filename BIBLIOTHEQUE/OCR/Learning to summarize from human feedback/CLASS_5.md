# Learning to summarize from human feedback (Not specified in the paper.)
Source: Learning to summarize from human feedback.md

## Core reasons
- The work proposes and evaluates training language models with human preference-based reward modeling and reinforcement learning, which is a training/alignment methodology rather than positional encoding or dimensional adaptation.
- The paper centers on misalignment between maximum-likelihood training and human-judged quality, motivating an alternative optimization objective and RL fine-tuning.

## Evidence extracts
- "In this work, we show that it is possible to significantly improve summary quality by training a model to optimize for human preferences. We collect a large, high-quality dataset of human comparisons between summaries, train a model to predict the human-preferred summary, and use that model as a reward function to fine-tune a summarization policy using reinforcement learning." (Abstract)
- "there is still a misalignment between this fine-tuning objective—maximizing the likelihood of human-written text—and what we care about—generating high-quality outputs as determined by humans." (Section 1 Introduction)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
