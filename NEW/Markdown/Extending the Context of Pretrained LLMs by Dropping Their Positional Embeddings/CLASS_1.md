# Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings (Not specified in the paper.)
Source: Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings.md

## Core reasons
- The paper critiques existing positional embeddings and RoPE scaling as blocking zero-shot long-context generalization in transformer LMs.
- The core contribution is DroPE, which changes positional encoding by removing positional embeddings after pretraining to extend context length.

## Evidence extracts
- "Second, over-reliance on this explicit positional information is also precisely what prevents test-time generalization to sequences of unseen length, even when using popular PE-scaling methods." (Section 1. Introduction)
- "Positional embeddings can be **removed after pretraining**, allowing LMs to generalize **zero-shot** to **unseen sequence lengths** without compromising their in-context performance after short recalibration on a fraction of the training tokens at the original context size." (Section 5. DroPE: Dropping positional embeddings after pretraining)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
