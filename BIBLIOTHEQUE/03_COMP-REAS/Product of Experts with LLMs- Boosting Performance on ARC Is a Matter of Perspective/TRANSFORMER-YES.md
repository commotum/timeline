# Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective (2025)
Source: Product of Experts with LLMs- Boosting Performance on ARC Is a Matter of Perspective.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s core method is explicitly built around large language models (LLMs), and the main reported ARC results are produced with those models.
- Auxiliary analysis identifies the concrete central models as Mistral-NeMo-Minitron-8B-Base and Llama 3B, both GPT/BERT/LLaMA-style Transformer families that materially use self-attention.

## Evidence
- "The Abstraction and Reasoning Corpus (ARC-AGI) poses a significant challenge for large language models (LLMs), exposing limitations in their abstract reasoning abilities." (Abstract, `Product of Experts with LLMs- Boosting Performance on ARC Is a Matter of Perspective.md`)
- "After evaluating various models, we identified **Mistral-NeMo-Minitron-8B-Base** (Sreenivas et al., 2024) as exhibiting the strongest performance in our experiments." (Section 5.2 quote reproduced in `TASK_MODEL_RATIO.md`)
- "Instead, we start out with our Llama 3B model pre-trained on ARC, which we then finetune again on 128000 Sudoku tasks." (Section 5.6 quote reproduced in `TASK_MODEL_RATIO.md`)
- "Extending-dimensions analysis markdown: MISSING" (Task prompt; file unavailable and therefore skipped)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence Transformer-family classification from abstract + TASK-DOMAINS/TASK-DOMAINS.csv/TASK_MODEL_RATIO.
Pass 2 (targeted source scan): skipped - not needed after Pass 1.
