# Keyword-Sort Pipeline Summary

Date: 2026-02-10

## Scope

This pipeline screens papers in:

- `BIBLIOTHEQUE/03_COMP-REAS`
- `BIBLIOTHEQUE/05_ML-FNDTNS`

using only `.md` and `.csv` sources (no PDFs/images in the classification logic).

Primary objective:

1. Decide whether a paper is transformer-family (or hybrid with transformer attention).
2. For transformer papers, classify task dimensionality (`1D`, `2D`, `3D`, `4D`, `multi-D`).
3. For `2D+` transformer papers, classify positional encoding (PE) method.

## Files That Implement The Pipeline

- `RUNNERS/keyword-sort/run_keyword_pipeline.py`
- `RUNNERS/keyword-sort/iterative_stage_refinement.py`

## Pipeline Stages

## Stage 1: Transformer Screening

Output:

- `RUNNERS/keyword-sort/transformer_screen_results.csv`
- `RUNNERS/keyword-sort/transformer_screen_summary.md`

Labels:

- `transformer_yes`
- `hybrid_transformer_yes`
- `transformer_no`
- `uncertain`

## Stage 2: Dimensionality Classification

Output:

- `RUNNERS/keyword-sort/transformer_task_dimensions_results.csv`
- `RUNNERS/keyword-sort/transformer_task_dimensions_summary.md`

Source priority:

1. `TASK-DOMAINS.csv` (primary)
2. `TASK-DOMAINS.md` (fallback)
3. OCR `.md` paper text (last fallback)

## Stage 3: Positional Encoding Classification

Output:

- `RUNNERS/keyword-sort/positional_encoding_results.csv`
- `RUNNERS/keyword-sort/positional_encoding_summary.md`
- `RUNNERS/keyword-sort/positional_encoding_uncertain.md`

Applied only to Stage-2 papers labeled `2D_only`, `3D_only`, `4D_only`, or `multi-D`.

## Stage 3.5: PE Priors For Known Backbones

Conservative title-level priors are stored in:

- `RUNNERS/keyword-sort/iterative_refinement_rules.json` (`stage3_pe_overrides`)

These reduce uncertainty for well-known backbone families when text evidence is sparse.

## Key Refinements Applied

1. Fixed Stage-1 false positives from Atari game name `Q*Bert` accidentally matching `bert`.
2. Improved Stage-1 title-audit logic so RL-only force-no overrides apply only when transformer signal is absent (`A_hits=0` and `B_hits=0`).
3. Enforced Stage-2 rule: do not downcast to `1D_only` by title when `TASK-DOMAINS.csv` already contains `2D/3D/4D`.
4. Expanded Stage-3 PE phrase coverage (for common wording like positional embeddings being added/summed) and guarded against cross-family contamination.
5. Added Stage-3.5 PE priors for known models/backbones (logged in refinement rules).

## Final Counts (Current CSV State)

From `transformer_screen_results.csv`:

- Total papers: `423`
- `transformer_yes`: `179`
- `hybrid_transformer_yes`: `50`
- `transformer_no`: `122`
- `uncertain`: `72`

From `transformer_task_dimensions_results.csv`:

- Transformer papers carried forward: `229`
- `1D_only`: `114`
- `2D_only`: `26`
- `3D_only`: `2`
- `multi-D`: `87`

From `positional_encoding_results.csv`:

- 2D+/multi-D papers processed: `115`
- `learned_absolute`: `16`
- `rope`: `6`
- `relative_position`: `6`
- `sinusoidal_absolute`: `3`
- `other_variant`: `11`
- `mixed`: `1`
- `alibi`: `1`
- `none_or_implicit`: `1`
- `unclear`: `70`

## What Was Cleaned Up

Top-level `RUNNERS/keyword-sort` now keeps only core scripts, plans, final CSV outputs, and summary docs.

Intermediate traces were moved to:

- `RUNNERS/keyword-sort/artifacts/traces/`

This includes large hit dumps (`hits_*.tsv`), candidate lists, and temporary OCR/task-domain list files.

## How To Re-Run

From repo root:

```bash
python RUNNERS/keyword-sort/run_keyword_pipeline.py
python RUNNERS/keyword-sort/iterative_stage_refinement.py
```

Notes:

- Running these commands regenerates outputs and intermediate traces at top-level.
- If you want the folder clean again, move intermediate traces back into `artifacts/traces/` after re-run.

## Current Caveats

1. `unclear=70` remains in Stage 3; these are largely papers where PE is inherited, unstated, or orthogonal to the paper's core contribution.
2. Stage-3.5 priors are useful but should be treated as assumptions unless confirmed by explicit text in the paper OCR or task files.
3. If taxonomy thresholds change, the authoritative source is always the three final CSV outputs.

