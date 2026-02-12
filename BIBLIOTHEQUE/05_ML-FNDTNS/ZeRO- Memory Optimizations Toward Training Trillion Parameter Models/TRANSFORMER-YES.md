# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2020)
Source: ZeRO- Memory Optimizations Toward Training Trillion Parameter Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The auxiliary analyses explicitly state that the evaluated models are GPT-2-like transformer models, making Transformer architectures central to the reported results.
- The abstract highlights ZeRO’s core demonstrated gains on GPT/T5-class language models; the extending-dimensions analysis file was unavailable (`MISSING`) but not required for a high-confidence decision.

## Evidence
- "The models presented in this section are GPT-2 [2] like transformer based models." (Section 10.1, `TASK-DOMAINS.md`)
- "In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism..." (Abstract, `ZeRO- Memory Optimizations Toward Training Trillion Parameter Models.md`)
- "As of May 12th, 2020, Turing-NLG is the largest model in the world with over 17B parameters." (Section 10.6, `TASK-DOMAINS.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence Transformer classification from the abstract and auxiliary files; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient.
