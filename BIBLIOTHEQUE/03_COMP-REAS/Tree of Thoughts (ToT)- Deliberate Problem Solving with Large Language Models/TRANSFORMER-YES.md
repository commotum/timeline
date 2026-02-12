# Tree of Thoughts: Deliberate Problem Solving with Large Language Models (2023)
Source: Tree of Thoughts (ToT)- Deliberate Problem Solving with Large Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract’s reported core results are produced with GPT-4, and ToT is an inference-time framework built on top of that LM rather than a separate non-attention architecture.
- Auxiliary analysis states one pre-trained LM instance covers all tasks and names GPT-4 as the default experimental model, which is a Transformer-family LLM.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract + auxiliary evidence is sufficient for a high-confidence decision.

## Evidence
- "For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4% of tasks, our method achieved a success rate of 74%." (Abstract, Tree of Thoughts (ToT)- Deliberate Problem Solving with Large Language Models.md:20)
- "No extra training is needed, just a pre-trained LM is sufficient." (Section 3) (TASK_MODEL_RATIO.md:9)
- "Unless otherwise stated, we perform experiments using a Chat Completion mode GPT-4" (Section 4) (TASK_MODEL_RATIO.md:11)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for TRANSFORMER-YES from abstract + TASK-DOMAINS/TASK-DOMAINS.csv/TASK_MODEL_RATIO; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence model-family evidence.
