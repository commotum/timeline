# Better & Faster Large Language Models via Multi-token Prediction (2024)
Source: a11504-2024.pdf

## Core reasons
- Frames training as predicting the next n tokens simultaneously with n output heads on a shared trunk, turning multi-token prediction into an auxiliary objective that boosts sample efficiency beyond next-token modeling (Abstract, Section 2).
- Demonstrates scalable empirical gains on code and language benchmarks and faster inference through self-speculative decoding, establishing the training objective as a foundation-level improvement rather than a niche acceleration trick (Section 3 and Section 3.2).

## Evidence extracts
- "Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk." (Abstract)
- "Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale." (Section 3.1)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
