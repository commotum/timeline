# The Lessons of Developing Process Reward Models in Mathematical Reasoning (Year not specified)
Source: The Lessons of Developing Process Reward Models....md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The central PRMs are built by initializing from Qwen2.5-Math-7B/72B-Instruct language models and adapting their LM head, indicating the main system is an LLM-based architecture family.
- Qwen2.5-Math-PRM-7B/72B are the paper’s primary trained/evaluated models; these are GPT-style LLM model lines (Transformer-family) rather than non-attention architectures.

## Evidence
- "Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in the reasoning processes." (Abstract, The Lessons of Developing Process Reward Models....md:14)
- "**Training Details** Our trained PRMs were initialized from the supervised fine-tuned Qwen2.5-Math-7B/72B-Instruct models (Yang et al., 2024c), where we replace the original language modeling head (used for next token prediction) with a scalar-value head, consisting of two linear layers." (Section 2.1 Training Details, The Lessons of Developing Process Reward Models....md:47)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract plus TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md in full; Extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - Minimal targeted scan of model/training lines confirmed the core PRMs are initialized from Qwen2.5-Math-7B/72B-Instruct models.
