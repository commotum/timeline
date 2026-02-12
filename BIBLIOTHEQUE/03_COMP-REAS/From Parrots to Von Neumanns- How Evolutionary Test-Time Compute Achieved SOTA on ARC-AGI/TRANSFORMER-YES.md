# From Parrots to Von Neumanns: How Evolutionary Test-Time Compute Achieved State-of-the-Art on ARC-AGI (2025)
Source: From Parrots to Von Neumanns- How Evolutionary Test-Time Compute Achieved SOTA on ARC-AGI.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s main systems are built around frontier LLMs (Claude Sonnet 3.5 and Grok-4) as the core generators/executors for the reported SOTA results.
- The auxiliary analyses show Grok-4 is central in production for both instruction generation and follower execution, so Transformer-family LLMs are material to the method.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract + available auxiliary files were sufficient for a high-confidence call.

## Evidence
- "My first system (2024) evolved Python functions using Claude Sonnet 3.5, achieving 53.6% on ARC-AGI-1. My second system (2025) evolved natural-language instructions using Grok-4, achieving 79.6% on ARC-AGI-1 and 29.4% on ARC-AGI-2." (Abstract, `From Parrots to Von Neumanns- How Evolutionary Test-Time Compute Achieved SOTA on ARC-AGI.md`)
- "A separate \"follower\" LLM (also Grok-4 in my production system) reads these instructions and applies them to each grid." (`TASK-DOMAINS.md`, Evidence section citing Section 5.2)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence decision; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already established a high-confidence decision.
