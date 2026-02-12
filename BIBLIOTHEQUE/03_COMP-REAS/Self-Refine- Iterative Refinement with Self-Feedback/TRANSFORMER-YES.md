# SELF-REFINE: Iterative Refinement with Self-Feedback (2023)
Source: Self-Refine- Iterative Refinement with Self-Feedback.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract identifies GPT-3.5 and GPT-4 LLMs as the models used for the main evaluation, and Self-Refine runs by repeatedly calling that LLM for generation, feedback, and refinement.
- Auxiliary analyses (TASK-DOMAINS.md, TASK-DOMAINS.csv, TASK_MODEL_RATIO.md) are consistent with a single central LLM-driven method; the extending-dimensions analysis file was unavailable (`MISSING`).

## Evidence
- "We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5 and GPT-4) LLMs." (Abstract, Self-Refine- Iterative Refinement with Self-Feedback.md)
- "Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider." (Abstract, Self-Refine- Iterative Refinement with Self-Feedback.md)
- "Number of trained model instances required to cover all tasks: 1 model." (TASK_MODEL_RATIO.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence classification from abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions analysis was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient to finalize.
