# Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents (Year not specified)
Source: Efficiently Allocating Test-Time Compute for LLM Agents.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s main method is an LLM-agent training pipeline and experiments explicitly use Llama-3.x models as the trained/evaluated agents; Llama-family models are Transformer-based self-attention architectures.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract and available auxiliary files already provide sufficient direct model-family evidence.

## Evidence
- "Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities." (Abstract, `Efficiently Allocating Test-Time Compute for LLM Agents.md`)
- "To understand baseline capabilities and the raw effect of planning frequency, we perform zero-shot evaluations using Llama-3.3-70B-Instruct (Grattafiori et al., 2024) on POGS and Crafter (100 seeds each)." (Section 4.3 quote in `TASK_MODEL_RATIO.md`)
- "**SFT Priming:** The Llama-3.1-8B model was fine-tuned on this data, aligning the SFT process with the target RL configuration." (Section 4.4 quote in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence Transformer classification from abstract plus model-family cues in auxiliary files.
Pass 2 (targeted source scan): skipped - not needed because Pass 1 was conclusive.
