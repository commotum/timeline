# Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability (2026)
Source: Teaching Models to Teach Themselves- Reasoning at the Edge of Learnability.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper's main teacher and student models are both Llama-3.2-3B-Instruct, a LLaMA-family Transformer LLM.
- The main reported results come from training and evaluating this Llama-based teacher-student setup, so Transformer architecture is central rather than peripheral.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files and targeted scan were sufficient to decide.

## Evidence
- "All experiments are conducted with Llama-3.2-3B-Instruct." (Teaching Models to Teach Themselves- Reasoning at the Edge of Learnability.md, Section 4.1, line 113)
- "Both the teacher and student are initialized from Llama-3.2-3B-Instruct." (Teaching Models to Teach Themselves- Reasoning at the Edge of Learnability.md, Section 4.2, line 119)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Abstract and auxiliary files indicated LLM-centric methodology, but did not by themselves provide explicit architecture naming at high confidence.
Pass 2 (targeted source scan): performed - Model/setup sections explicitly identify Llama-3.2-3B-Instruct as the core model used for main experiments.
