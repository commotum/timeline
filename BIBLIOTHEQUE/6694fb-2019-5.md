# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2020)
Source: 6694fb-2019.pdf

## Core reasons
- The paper’s main contribution is a model-parallel training method for transformers, not a new positional encoding or higher-dimensional representation.
- It focuses on scaling training under memory constraints and system efficiency, which is a training methodology contribution.

## Evidence extracts
- "We implement a simple and efﬁcient model parallel approach by making only a few targeted modiﬁcations to an existing PyTorch transformer implementation." (p. 2)
- "However, these techniques have one fundamental limitation in the problem size they can tackle: the model must ﬁt entirely on one worker." (p. 3)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
