# Generating Long Sequences with Sparse Transformers (Not specified in the paper.)
Source: Generating Long Sequences with Sparse Transformers.md

## Core reasons
- The paper's main contribution is a Transformer architecture modification using sparse, factorized attention to reduce complexity and enable long-sequence modeling, rather than positional encoding or dataset creation.
- It focuses on efficiency and training/architecture changes for Transformers (sparse attention, deep networks, recomputation), which fits general ML modeling advances.

## Evidence extracts
- "In this paper we introduce sparse factorizations of the attention matrix which reduce this to  $O(n\sqrt{n})$ . We also introduce a) a variation on architecture and initialization to train deeper networks, b) the recomputation of attention matrices to save memory, and c) fast attention kernels for training." (Abstract)
- "The main contribution of this work is to introduce several sparse factorizations of the attention matrix, which scale as  $O(n\sqrt[p]{n})$  with the sequence length without sacrificing performance." (Section 1. Introduction)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
