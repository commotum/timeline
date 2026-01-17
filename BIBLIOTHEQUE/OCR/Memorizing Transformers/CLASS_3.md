# Memorizing Transformers (2022)
Source: 87ef1d-2022.pdf

## Core reasons
- Introduces an inference-time memory capability for language models, letting them read and memorize new data to acquire knowledge immediately.
- Adds a kNN-augmented attention mechanism that performs approximate k-nearest-neighbor search over an external memory, changing how computation is performed.

## Evidence extracts
- "We instead envision language models that can simply read and memorize new data at inference time, thus acquiring new knowledge immediately. In this work, we extend language models with the ability to memorize the internal representations of past inputs." (p. 1)
- "Like all of the other layers, it uses standard dense self-attention on the local context, which is the input subsequence for the current training step. Unlike the other layers, however, it also does an approximate k-nearest-neighbor search into the external memory." (p. 4)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
