# Training language models to follow instructions with human feedback (2022)
Source: Training language models to follow instructions with human feedback (InstructGPT - RLHF pipeline).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states the central method fine-tunes GPT-3 (SFT + RLHF) to produce InstructGPT, so the main model family is GPT-3.
- The auxiliary analysis explicitly identifies GPT-3 architecture signals as central model-family evidence, consistent with Transformer-based LLM blocks.
- The extending-dimensions file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning." (Abstract, Training language models to follow instructions with human feedback (InstructGPT - RLHF pipeline).md:17)
- "All model architectures use the GPT-3 architecture" (TASK-DOMAINS.md:28)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for TRANSFORMER-YES from abstract + TASK-DOMAINS.md/TASK-DOMAINS.csv/TASK_MODEL_RATIO.md; extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient for high-confidence classification.
