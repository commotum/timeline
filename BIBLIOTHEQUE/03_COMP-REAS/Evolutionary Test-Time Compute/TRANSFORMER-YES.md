# How I came in first on ARC-AGI-Pub using Sonnet 3.5 with Evolutionary Test-time Compute (2024)
Source: Evolutionary Test-Time Compute.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: medium
Basis: abstract-aux-only

## Why
- The method’s core engine is repeated use of Claude Sonnet 3.5 (an LLM family cue consistent with Transformer-based models) to generate and revise candidate solutions.
- Auxiliary analyses indicate one central model instance (Sonnet 3.5) for the main task rather than Transformers appearing only as peripheral baselines.
- The extending-dimensions analysis file was unavailable (`MISSING`), so the decision is based on the abstract-equivalent opening text and the three available auxiliary files.

## Evidence
- "After lots of experimenting, I got a record of 53.6% on the public leaderboard using Sonnet 3.5." (Evolutionary Test-Time Compute.md, opening section/abstract-equivalent)
- "My approach works by having Sonnet 3.5 generate a bunch of Python transform functions" (Evolutionary Test-Time Compute.md, opening section/abstract-equivalent)
- "I compared two architectures across 60 training challenges, each using 200 LLM calls with Sonnet 3.5 and identical system prompts and examples:" (TASK_MODEL_RATIO.md, item 2 quote from source)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence to decide from abstract-equivalent opening text + available auxiliary files.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; no additional body scan needed.
