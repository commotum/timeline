# Generative Verifiers: Reward Modeling as Next-Token Prediction (Year not specified)
Source: Generative Verifiers- Reward Modeling as Next-Token Prediction.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract presents GenRM as an LLM-based verifier trained with next-token prediction and reports Gemma-based verifiers as the main model family used for results.
- The auxiliary files are consistent with Gemma/LLM-centric modeling, and the expected extending-dimensions file was unavailable (`MISSING`) but not required for a confident decision.

## Evidence
- "To overcome this limitation, we instead propose training verifiers using the ubiquitous next-token prediction objective, jointly on verification and solution generation." (Abstract, `Generative Verifiers- Reward Modeling as Next-Token Prediction.md`, line 9)
- "We demonstrate that when using Gemma-based verifiers on algorithmic and grade-school math reasoning tasks, GenRM outperforms discriminative verifiers and LLM-as-a-Judge..." (Abstract, `Generative Verifiers- Reward Modeling as Next-Token Prediction.md`, line 9)
- "For solution generation as well as LLM-as-a-Judge, we use Gemma 2B for algorithmic tasks and Gemini 1.0 Pro [Team et al., 2023] for GSM8K." (`TASK-DOMAINS.md`, Models evidence quote)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence decision; `EXTENDING-DIMENSIONS.md` was unavailable (`MISSING`) and skipped.
Pass 2 (targeted source scan): skipped - pass 1 already provided high-confidence architecture/model-family evidence.
