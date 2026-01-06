# Adaptive Attention Span in Transformers (Not specified in the paper.)
Source: Adaptive Attention Span in Transformers.md

## Core reasons
- Introduces an adaptive self-attention span mechanism that changes how computation is performed per head rather than altering positional encoding.
- Addresses the quadratic compute/memory scaling of standard self-attention by learning span limits to extend context with controlled cost.

## Evidence extracts
- "We propose a novel self-attention mechanism that can learn its optimal attention span." (Abstract)
- "While this layer allows for information to propagate across long distances, it has a computational and memory cost that scales quadratically with the size of the input sequence." (1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
