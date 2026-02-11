# ARCMEMO: ABSTRACT REASONING COMPOSITION WITH LIFELONG LLM MEMORY (Year not specified)
Source: TASK-DOMAINS.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: medium
Basis: hint-only

## Why
- The paper’s primary system is explicitly built around an LLM for solving ARC tasks, with memory read/write added at inference time.
- The evaluated core model is OpenAI `o4-mini`, which is a GPT-family LLM and therefore a Transformer-style architecture in this context.

## Evidence
- "we experiment primarily with OpenAI's o4-mini." (TASK_MODEL_RATIO.md, quote sourced from Section 4, Models)
- "enabling test-time continual learning without weight updates." (TASK_MODEL_RATIO.md, quote sourced from Abstract)

## Pass accounting
Pass 0 (hint-first): performed - Hints identify the central model as an LLM (`o4-mini`), sufficient for a Transformer-YES decision.
Pass 1 (source triage): skipped - High-confidence hint evidence was sufficient.
Pass 2 (source deep dive): skipped - Not needed after Pass 0.
