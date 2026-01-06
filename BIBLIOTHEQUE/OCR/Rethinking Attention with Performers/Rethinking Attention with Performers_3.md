# Rethinking Attention with Performers (Not specified in the paper.)
Source: Rethinking Attention with Performers.md

## Core reasons
- Proposes a new attention computation mechanism (FAVOR+ / Performers) to approximate softmax attention with linear time and space, changing how attention is computed.
- Motivated by the quadratic cost of standard attention and offers a scalable computation alternative rather than positional encoding or domain lifting.

## Evidence extracts
- "We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity" (Abstract)
- "Transformers rely on a trainable attention mechanism that identifies complex dependencies between the elements of each input sequence. Unfortunately, the regular Transformer scales quadratically with the number of tokens L in the input sequence" (Section 1 Introduction and related work)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
