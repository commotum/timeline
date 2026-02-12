# StoryMem: Multi-shot Long Video Storytelling with Memory (2025)
Source: StoryMem- Multi-shot Long Video Storytelling with Memory.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes memory injection into a video diffusion backbone using "negative RoPE shifts," which is a Transformer positional encoding mechanism.
- The auxiliary model-ratio analysis explicitly states LoRA fine-tuning is applied to linear layers in "DiT blocks," indicating a Transformer-based diffusion architecture is central to the method.

## Evidence
- "The stored memory is then injected into single-shot video diffusion models via latent concatenation and negative RoPE shifts with only LoRA fine-tuning." (Abstract, StoryMem- Multi-shot Long Video Storytelling with Memory.md:15)
- "We finetune it using a rank-128 LoRA applied to all linear layers in the DiT blocks, adding  $\sim$ 0.7B active parameters." (TASK_MODEL_RATIO.md, item 2 with citation to Section 4.1)
- "Extending-dimensions analysis markdown" was unavailable because its path resolved to `MISSING`. (Input specification)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision from abstract + TASK_MODEL_RATIO/TASK-DOMAINS files.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; no additional sections were needed.
