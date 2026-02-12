# STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning (2022)
Source: STaR- Self-Taught Reasoner (Bootstrapping Reasoning With Reasoning).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper’s core experimental model is GPT-J, and the source explicitly states GPT-J is a decoder-only Transformer with multi-head attention.
- STaR is evaluated by prompting and fine-tuning that GPT-J model across the main tasks, so Transformer-style self-attention is central to the reported results.

## Evidence
- "We used GPT-J as our base language model, and the fine-tuning script from the GPT-J repository [26]." (STaR- Self-Taught Reasoner (Bootstrapping Reasoning With Reasoning).md, Section 4.1 Experimental Protocol, line 108)
- "GPT-J is a 28-layer decoder-only transformer, with an embedding size of 1024, 16 attention heads of dimension 256, and an FFN hidden layer of size 16384." (STaR- Self-Taught Reasoner (Bootstrapping Reasoning With Reasoning).md, Appendix H Hyperparameters, line 1096)
- "Generating step-by-step \"chain-of-thought\" rationales improves language model performance on complex reasoning tasks like mathematics or commonsense question-answering." (STaR- Self-Taught Reasoner (Bootstrapping Reasoning With Reasoning).md, Abstract, line 10)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md in full; evidence indicated an LLM-centric method but did not by itself give a direct architecture statement. Extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): performed - Searched the source for architecture cues and found explicit confirmation that the central model (GPT-J) is a decoder-only Transformer with attention heads.
