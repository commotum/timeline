# Self-Consistency Improves Chain of Thought Reasoning in Language Models (2022)
Source: Self-Consistency Improves Chain of Thought Reasoning in Language Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract frames the method as operating on pre-trained large language models, and the paper explicitly states evaluation on "four transformer-based language models," making Transformer architecture central to the reported results.
- Auxiliary analyses (`TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, `TASK_MODEL_RATIO.md`) consistently describe a single language-model-centric setup across all tasks, with no alternative non-attention architecture as the core method.
- The extending-dimensions analysis file was unavailable (`MISSING`), but Pass 1 evidence is already direct and sufficient.

## Evidence
- "Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks." (Abstract, `Self-Consistency Improves Chain of Thought Reasoning in Language Models.md`:9)
- "**Language models and prompts.** We evaluate self-consistency over four transformer-based language models with varying scales:" (Section 3.1, `Self-Consistency Improves Chain of Thought Reasoning in Language Models.md`:74)
- "Self-consistency also differs from a typical ensemble approach where multiple models are trained and the outputs from each model are aggregated, it acts more like a \"self-ensemble\" that works on top of a *single* language model." (`TASK_MODEL_RATIO.md`:17)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence YES; explicit Transformer-model statement found; `MISSING` extending-dimensions file unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 already provided direct architecture evidence.
