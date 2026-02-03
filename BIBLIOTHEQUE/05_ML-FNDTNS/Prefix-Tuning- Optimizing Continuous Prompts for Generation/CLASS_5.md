# Prefix-Tuning: Optimizing Continuous Prompts for Generation (Not specified in the paper.)
Source: Prefix-Tuning- Optimizing Continuous Prompts for Generation.md

## Core reasons
- The paper proposes prefix-tuning as a parameter-efficient alternative to fine-tuning by keeping the pretrained model fixed and optimizing a small continuous prefix, which is a training methodology contribution rather than a positional encoding or dimensionality change.
- The method centers on how to adapt pretrained language models with minimal task-specific parameters for generation tasks, fitting ML training principles and parameter-efficient adaptation.

## Evidence extracts
- "we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix)." (Abstract)
- "the language model parameters  $\phi$  are fixed and the prefix parameters  $\theta$  are the only trainable parameters." (Section 4.2 Method)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
