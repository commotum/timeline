# Language Models are Few-Shot Learners (Year not specified)
Source: Language Models are Few-Shot Learners.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract identifies the central model as GPT-3; GPT-family models are Transformer-based autoregressive language models that rely on self-attention blocks.
- Auxiliary files consistently show GPT-3 as the single model used across all evaluated tasks; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters" (Language Models are Few-Shot Learners.md, Abstract)
- "For all tasks, GPT-3 is applied without any gradient updates or fine-tuning" (TASK_MODEL_RATIO.md, item 2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence GPT-3-centered Transformer classification.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; no additional source scan needed.
