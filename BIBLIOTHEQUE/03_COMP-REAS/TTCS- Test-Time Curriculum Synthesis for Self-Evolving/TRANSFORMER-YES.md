# TTCS: Test-Time Curriculum Synthesis for Self-Evolving (Year not specified)
Source: TTCS- Test-Time Curriculum Synthesis for Self-Evolving.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines TTCS as test-time adaptation for "large language models (LLMs)", and the method’s core solver/synthesizer are initialized from a pretrained LLM.
- The experiments are run on Qwen LLM backbones (Qwen2.5/Qwen3), which are Transformer-family language models; TTCS directly trains/adapts these backbones.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus provided auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "Test-Time Training offers a promising way to improve the reasoning ability of large language models (LLMs) by adapting the model using only the test questions." (Abstract, `TTCS- Test-Time Curriculum Synthesis for Self-Evolving.md`)
- "To demonstrate the scalability of **TTCS**, we conduct experiments on three base pretrained models: Qwen2.5-Math-1.5B, Qwen2.5-Math-7B [25] and Qwen3-4B-Base [40]." (Section 5.1 Experimental Setting, `TTCS- Test-Time Curriculum Synthesis for Self-Evolving.md`)
- "TTCS initializes two policies from the same pretrained model: a question synthesizer and a reasoning solver." (Evidence section, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision from abstract + TASK-DOMAINS/TASK-DOMAINS.csv/TASK_MODEL_RATIO; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already sufficient.
