# Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models (Year not specified)
Source: Plan-and-Solve Prompting.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s main method (Plan-and-Solve prompting) is evaluated using GPT-3 as the backbone model, and GPT-3 is a Transformer-family LLM.
- Auxiliary analysis confirms a single central model instance (GPT-3/text-davinci-003) across all tasks; Transformer use is core, not peripheral.
- The Extending-dimensions analysis file was unavailable (`MISSING`), but the abstract and auxiliary files provide sufficient direct model evidence.

## Evidence
- "The experimental results over GPT-3 show that our proposed zero-shot prompting consistently outperforms Zero-shot-CoT across all datasets by a large margin" (Abstract, Plan-and-Solve Prompting.md)
- "Following Auto-CoT (Zhang et al., 2022), we use the public GPT-3 (Brown et al., 2020) (175B) as the backbone language model" (Section 3.3 Implementations, TASK_MODEL_RATIO.md)
- "We report the results using text-davinci-003 engine for GPT-3 in the main paper." (Section 3.3 Implementations, TASK_MODEL_RATIO.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence YES decision from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Not needed because Pass 1 already identifies GPT-3 as the central backbone model.
