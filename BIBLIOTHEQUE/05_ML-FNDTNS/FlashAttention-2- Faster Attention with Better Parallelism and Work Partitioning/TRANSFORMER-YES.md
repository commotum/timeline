# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023)
Source: FlashAttention-2- Faster Attention with Better Parallelism and Work Partitioning.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly frames the work as improving Transformer attention, not as a peripheral baseline comparison.
- Auxiliary analyses and quoted evaluation context focus on attention kernels and end-to-end GPT-style model training.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the available Pass 1 sources are already decisive.

## Evidence
- "Scaling Transformers to longer sequence lengths has been a major problem in the last several years" (FlashAttention-2- Faster Attention with Better Parallelism and Work Partitioning.md, Abstract)
- "The attention layer is the main bottleneck in scaling to longer sequences" (FlashAttention-2- Faster Attention with Better Parallelism and Work Partitioning.md, Abstract)
- "When used end-to-end to train GPT-style models" (TASK_MODEL_RATIO.md, Section 4 quote)
- "Extending-dimensions analysis markdown: MISSING" (Input specification; file unavailable)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-YES from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md.
Pass 2 (targeted source scan): skipped - Not needed because Pass 1 was already conclusive.
