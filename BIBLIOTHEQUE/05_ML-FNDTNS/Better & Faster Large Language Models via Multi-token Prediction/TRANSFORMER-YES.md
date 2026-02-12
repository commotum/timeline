# Better & Faster Large Language Models via Multi-token Prediction (Year not specified)
Source: Better & Faster Large Language Models via Multi-token Prediction.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the work as a modification to GPT/Llama-family LLM training, and GPT/Llama are Transformer decoder architectures that materially rely on self-attention.
- Auxiliary files are consistent with an autoregressive LLM setup, and although the extending-dimensions analysis file was unavailable (`MISSING`), the available evidence is sufficient for a high-confidence decision.

## Evidence
- "Large language models such as GPT and Llama are trained with a next-token prediction loss." (Abstract, Better & Faster Large Language Models via Multi-token Prediction.md)
- "multi-token prediction instructs the LLM to predict the *n* future tokens from each position in the training corpora, all at once and in parallel (Qi et al., 2020)." (Evidence section, TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-YES.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; extending-dimensions file was unavailable (`MISSING`).
