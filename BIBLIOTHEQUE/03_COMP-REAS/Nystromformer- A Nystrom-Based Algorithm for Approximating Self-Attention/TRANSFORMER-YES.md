# Nyströmformer: A Nyström-based Algorithm for Approximating Self-Attention (2021)
Source: Nystromformer- A Nystrom-Based Algorithm for Approximating Self-Attention.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines Nyströmformer as an approximation of standard self-attention in Transformers, so Transformer-style self-attention is the central mechanism.
- Auxiliary analyses align with this framing across tasks; the extending-dimensions analysis file was unavailable (`MISSING`) but Pass 1 evidence was already decisive.

## Evidence
- "To address this limitation, we propose Nyströmformer – a model that exhibits favorable scalability as a function of sequence length. Our idea is based on adapting the Nyström method to approximate standard self-attention with O(n) complexity." (Abstract, `Nystromformer- A Nystrom-Based Algorithm for Approximating Self-Attention.md`:9)
- "Attention and state dynamics are inferred as static and direct because the model applies Transformer self-attention over the provided input sequences." (`TASK-DOMAINS.md`, Summary, line 22)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence TRANSFORMER-YES using the abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions analysis was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
