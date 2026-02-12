# What matters when building vision-language models? (2024)
Source: What matters when building vision-language models- (Idefics2).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract directly says VLM progress is driven by "large language models and vision transformers," and Idefics2 is presented as the paper's central VLM.
- The auxiliary analysis describes Idefics2 as using an autoregressive architecture with attention dynamics, consistent with Transformer-family multimodal models.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files are sufficient for a high-confidence decision.

## Evidence
- "The growing interest in vision-language models (VLMs) has been driven by improvements in large language models and vision transformers." (Abstract, What matters when building vision-language models- (Idefics2).md)
- "The paper describes Idefics2 as a multimodal text-generating VLM covering broad vision-language tasks plus text-only instruction data. Supported tasks span visual QA, OCR/document/chart/table tasks, reasoning (including geometry), counting, captioning, difference spotting, screenshot-to-code, math/arithmetic, and chat dialogue generation. Inputs cover 2D visual objects and 1D token sequences, while outputs are text tokens (1D). Based on explicit maximum sequence lengths and resolution limits, most interfaces are Capped; chat is Open due ongoing multi-turn interaction, with Static attention and Direct state inferred from the described autoregressive architecture." (Summary, TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence YES from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
