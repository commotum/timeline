# Recursive Language Models (Year not specified)
Source: Recursive Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The central method is a recursive scaffold over frontier LLMs, and main results are produced by GPT-5 and Qwen3-Coder model calls (GPT-style Transformer-family LMs).
- The source explicitly references Transformer neural networks in the core method description; the Extending-dimensions analysis file was unavailable (`MISSING`) and was skipped.

## Evidence
- "We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling." (Abstract, `Recursive Language Models.md`)
- "The key insight is that long prompts should not be fed into the neural network (e.g., Transformer) directly but should instead be treated as *part of the environment that the LLM can symbolically interact with*." (§1 Introduction, `Recursive Language Models.md`)
- "We evaluate RLMs using a frontier closed model (GPT-5; OpenAI 2025) and a frontier open model (Qwen3-Coder-480B-A35B; Team 2025)" (§1 Introduction, `Recursive Language Models.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; Extending-dimensions file was unavailable (`MISSING`); likely Transformer-based but needed explicit architecture cue.
Pass 2 (targeted source scan): performed - Confirmed explicit Transformer mention and that core evaluated models are GPT-5/Qwen3-Coder.
