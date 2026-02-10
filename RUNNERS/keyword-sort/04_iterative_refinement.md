# Iterative Refinement

## Goal
Run stage-by-stage loops to find obvious title-level mistakes, patch logic/overrides, rerun, and stop when stable.

## Loop
1. Run stage output.
2. Audit title-level errors.
3. Add targeted fixes/overrides.
4. Rerun until stable.

## Key Improvements Applied
- Stage 1: fixed `Q*Bert` -> `bert` false positives.
- Stage 1: RL-only force-no now requires no transformer signal (`A_hits=0` and `B_hits=0`).
- Stage 2: do not force `1D_only` if `TASK-DOMAINS.csv` has `2D/3D/4D`.
- Stage 3: expanded PE phrase coverage with extra guards to reduce cross-family contamination.

## Current Manual Overrides
- Stage 1 force-yes:
  - `Advancing Process Verification for LLM Reasoning`
- Stage 1 force-no:
  - none
- Stage 2 force-1D:
  - none

## Stage 3.5 PE Priors (Known Backbone/Family)
- `ALBEF- Align Before Fuse` -> `learned_absolute` (medium)
- `BEiT- BERT Pre-Training of Image Transformers` -> `learned_absolute` (medium)
- `BLIP- Bootstrapping Language-Image Pre-training` -> `learned_absolute` (medium)
- `BLIP-2- Bootstrapping Language-Image Pre-training with Frozen Models` -> `learned_absolute` (medium)
- `Image as a Foreign Language- BEiT Pretraining for All Vision and Vision-Language Tasks` -> `learned_absolute` (medium)
- `Learning Transferable Visual Models From Natural Language Supervision` -> `learned_absolute` (medium)
- `Masked Autoencoders Are Scalable Vision Learners (MAE)` -> `sinusoidal_absolute` (high)
- `Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN)` -> `learned_absolute` (low)
- `Sigmoid Loss for Language Image Pre-Training` -> `learned_absolute` (medium)
- `Training data-efficient image transformers & distillation through attention` -> `learned_absolute` (medium)

## Stable End State
- Stage 1 total: `423`
- Stage 2 total: `229`
- Stage 3 total: `115`
- Stage 3 unclear: `70`

## Script Location
- `scripts/run_keyword_pipeline.py`
- `scripts/iterative_stage_refinement.py`
