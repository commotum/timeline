# Iterative Refinement Report

This run performs stage-by-stage loops:
1. run stage
2. audit obvious title-level mistakes
3. investigate using current evidence fields and apply improvement overrides
4. repeat until stable or max iterations

## Stage 1 Loop
- Iteration 1: title-level issues found = 0
  - Stable.
## Stage 2 Loop
- Iteration 1: title-level issues found = 0
  - Stable.
## Stage 3 Loop
- Iteration 1: title-level issues found = 7
  - Added PE prior override: `ALBEF- Align Before Fuse` -> learned_absolute (medium). Cause: known backbone/title prior for PE exists
  - Added PE prior override: `BLIP- Bootstrapping Language-Image Pre-training` -> learned_absolute (medium). Cause: known backbone/title prior for PE exists
  - Added PE prior override: `BLIP-2- Bootstrapping Language-Image Pre-training with Frozen Models` -> learned_absolute (medium). Cause: known backbone/title prior for PE exists
  - Added PE prior override: `Learning Transferable Visual Models From Natural Language Supervision` -> learned_absolute (medium). Cause: known backbone/title prior for PE exists
  - Added PE prior override: `Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN)` -> learned_absolute (low). Cause: known backbone/title prior for PE exists
  - Added PE prior override: `Sigmoid Loss for Language Image Pre-Training` -> learned_absolute (medium). Cause: known backbone/title prior for PE exists
  - Added PE prior override: `Training data-efficient image transformers & distillation through attention` -> learned_absolute (medium). Cause: known backbone/title prior for PE exists
- Iteration 2: title-level issues found = 0
  - Stable.
## Final Counts
- Step1: total=423
  - hybrid_transformer_yes: 50
  - transformer_no: 122
  - transformer_yes: 179
  - uncertain: 72
- Step2: total=229
  - 1D_only: 114
  - 2D_only: 26
  - 3D_only: 2
  - multi-D: 87
- Step3: total=115
  - alibi: 1
  - learned_absolute: 16
  - mixed: 1
  - none_or_implicit: 1
  - other_variant: 11
  - relative_position: 6
  - rope: 6
  - sinusoidal_absolute: 3
  - unclear: 70
