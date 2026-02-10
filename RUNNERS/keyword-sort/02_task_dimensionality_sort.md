# Stage 2: Task Dimensionality Sort

## Goal
For Stage-1 transformer papers, classify model task dimensionality as `1D_only`, `2D_only`, `3D_only`, `4D_only`, or `multi-D`.

## Method
Source priority is strict:
1. `TASK-DOMAINS.csv` (primary)
2. `TASK-DOMAINS.md` (fallback)
3. OCR `.md` paper text (last fallback)

Dimension parsing uses explicit `1D/2D/3D/4D` markers and coordinate patterns.

## Output
- `transformer_task_dimensions_results.csv`

## Final Counts
- total: `229`
- `1D_only`: `114`
- `2D_only`: `26`
- `3D_only`: `2`
- `multi-D`: `87`

## Source Usage
- `task_csv_primary`: `227`
- `task_md_fallback`: `1`
- `ocr_fallback`: `1`

## Notes
- Title-based 1D overrides are blocked when `TASK-DOMAINS.csv` already shows `2D/3D/4D`.
