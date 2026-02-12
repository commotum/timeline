# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models (Year not specified)
Source: Self-Distilled Reasoner- On-Policy Self-Distillation for Large Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the method around a single "large language model (LLM)" acting as teacher and student, so the central model family is modern LLM architecture rather than non-attention baselines.
- Auxiliary analysis ties experiments to Qwen3 LLM variants (1.7B/4B/8B) and token-level autoregressive generation; this is consistent with Transformer-based decoder LLMs being the core model used for results.
- The Extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files are sufficient for a high-confidence decision.

## Evidence
- "Knowledge distillation improves large language model (LLM) reasoning by compressing the knowledge of a teacher LLM to train smaller LLMs." (Abstract, `Self-Distilled Reasoner- On-Policy Self-Distillation for Large Language Models.md`)
- "we introduce *On-Policy* Self-Distillation (OPSD), a framework where a single model acts as both teacher and student by conditioning on different contexts." (Abstract, `Self-Distilled Reasoner- On-Policy Self-Distillation for Large Language Models.md`)
- "At each position n, they induce *next-token* distributions over  $y_n \in \mathcal{V}$  conditioned on the same student prefix" (Evidence section, `TASK-DOMAINS.md`)
- "we introduce *On-Policy* Self-Distillation (OPSD), a framework where a single model acts as both teacher and student by conditioning on different contexts." (Verbatim evidence item 2, `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence Transformer-family classification from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Not needed after Pass 1.
