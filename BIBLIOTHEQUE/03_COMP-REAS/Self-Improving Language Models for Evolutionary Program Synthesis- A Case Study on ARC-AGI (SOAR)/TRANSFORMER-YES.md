# Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (2025)
Source: Self-Improving Language Models for Evolutionary Program Synthesis- A Case Study on ARC-AGI (SOAR).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The core SOAR method repeatedly uses and fine-tunes an LLM for both sampling and refinement, so the main results depend on that model family.
- Targeted source cues name Qwen and Mistral backbones (Transformer LLM families), indicating self-attention is materially part of the central model.
- The Extending-dimensions analysis file was unavailable (`MISSING`), so the decision uses the abstract plus available auxiliary files and targeted source cues.

## Evidence
- "an evolutionary search that uses an LLM to sample and refine candidate solutions" (Abstract, `Self-Improving Language Models for Evolutionary Program Synthesis- A Case Study on ARC-AGI (SOAR).md:7`)
- "We evaluate SOAR in combination with LLMs from the *Qwen-2.5-Coder* series (7B, 14B, 32B)" (Section 3.4, Implementation details, `Self-Improving Language Models for Evolutionary Program Synthesis- A Case Study on ARC-AGI (SOAR).md:136`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - abstract and all available auxiliary files were read fully (`TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, `TASK_MODEL_RATIO.md`); evidence indicated an LLM-centric method, and Extending-dimensions was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - scanned for architecture cues and found explicit Qwen/Mistral LLM backbone usage, confirming a Transformer-based central model.
