# Transformer Language Models without Positional Encodings Still Learn Positional Information (Not specified in the paper.)
Source: Transformer Language Models without Positional Encodings Still Learn Positional Information.md

## Core reasons
- The paper's main contribution is an empirical and probing analysis showing that transformers without explicit positional encodings still learn and use positional information, not a new encoding method.
- It provides a mechanistic hypothesis about causal attention and tests it via comparisons to masked language models, focusing on model behavior and principles.

## Evidence extracts
- "However, we show that LMs without any explicit positional encoding are still competitive with standard models, and that this phenomenon is robust across different datasets, model sizes, and sequence lengths. Probing experiments reveal that such models acquire an implicit notion of absolute positions throughout the network" (Abstract)
- "Overall, the probe reveals that the NoPos models learn an implicit notion of absolute positions." (Section 5 Analysis)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
