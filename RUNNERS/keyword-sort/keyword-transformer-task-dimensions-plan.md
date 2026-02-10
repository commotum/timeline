# Plan 2: Transformer Paper Task-Dimension Identification

## Goal
Starting from:
- `RUNNERS/keyword-sort/transformer_screen_results.csv`

identify dimensionality of tasks performed by Transformer-family papers:
- `1D`
- `2D`
- `3D`
- `4D`
- `multi-D` (more than one dimension category)

Use:
- OCR main paper markdown (`<paper_dir>/<paper_dir_name>.md`) as primary evidence.
- Runner-generated task files (`TASK-DOMAINS.csv`, `TASK-DOMAINS.md`) as confirmation.

## Inputs
- `RUNNERS/keyword-sort/transformer_screen_results.csv`
- Paper folders under:
  - `BIBLIOTHEQUE/03_COMP-REAS`
  - `BIBLIOTHEQUE/05_ML-FNDTNS`
- Inside each paper folder:
  - `<paper_name>.md` (OCR main text)
  - `TASK-DOMAINS.csv` (if present)
  - `TASK-DOMAINS.md` (if present)

## Counting Policy
- Include only papers where label in `transformer_screen_results.csv` is:
  - `transformer_yes`
  - `hybrid_transformer_yes`
- `uncertain` papers from step 1 are excluded by default and can be reviewed later.

## Dimension Policy (Important)
- Dimension is the **task domain dimension**, not model hidden-state size.
- Flattening a 2D/3D input into token sequences does **not** convert task dimension to 1D.
- Do not use representation dimensionality claims (for example latent PR/dimensionality) as task dimension evidence.
- Use natural task structure:
  - 1D: tokens, text/audio/time-series sequences
  - 2D: images/grids/tables/boards
  - 3D: `(x,y,z)` spatial volumes or `(x,y,t)` spatiotemporal planes/video
  - 4D: `(x,y,z,t)` explicit 3D space over time

## Keyword Sets

### 1D Cues
- `1D`, `one-dimensional`
- `sequence`, `token sequence`, `time series`, `audio waveform`
- `language modeling`, `autoregressive text`

### 2D Cues
- `2D`, `two-dimensional`
- `image`, `pixel`, `patch`, `grid`, `table`, `board`
- `maze`, `sudoku`, `(x, y)`

### 3D Cues
- `3D`, `three-dimensional`
- `point cloud`, `voxel`, `volume`, `mesh`
- `video`, `(x, y, z)`, `(x, y, t)`

### 4D Cues
- `4D`, `four-dimensional`
- `(x, y, z, t)`, `spatiotemporal volume`, `space-time volume`

## Evidence Priority
1. OCR method/architecture/task/experiment sections (primary)
2. `TASK-DOMAINS.csv` (`in_dimension`, `out_dimension`) (confirmation)
3. `TASK-DOMAINS.md` table/summary/evidence (secondary confirmation)

## Guardrails
- Ignore references/bibliography-only hits.
- Ignore related-work-only hits unless supported in method/task sections.
- Require at least one line-cited quote from OCR for final dimension label.
- If OCR and task files conflict, prefer OCR and set confidence `medium` or `low`.

## Workflow

### Step 1: Build Transformer Paper List From Step-1 CSV
Create:
- `RUNNERS/keyword-sort/transformer_paper_dirs.txt`

Use robust CSV parsing:

```bash
python - <<'PY'
import csv
src = "RUNNERS/keyword-sort/transformer_screen_results.csv"
out = "RUNNERS/keyword-sort/transformer_paper_dirs.txt"
keep = {"transformer_yes", "hybrid_transformer_yes"}
with open(src, newline="", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    r = csv.DictReader(f)
    for row in r:
        label = (row.get("label") or "").strip().lower()
        paper_dir = (row.get("paper_dir") or "").strip()
        if label in keep and paper_dir:
            g.write(paper_dir + "\n")
PY
```

### Step 2: Resolve OCR and Task Files
For each paper dir:
- OCR: `<dir>/<basename(dir)>.md`
- Optional confirmations:
  - `<dir>/TASK-DOMAINS.csv`
  - `<dir>/TASK-DOMAINS.md`

Create file manifests:
- `transformer_ocr_files.txt`
- `transformer_taskdomain_csv_files.txt`
- `transformer_taskdomain_md_files.txt`
- `transformer_missing_ocr.txt`

### Step 3: Fast Dimension Keyword Scan (Low Token)
Create hit files:
- `hits_dim_1d.tsv`
- `hits_dim_2d.tsv`
- `hits_dim_3d.tsv`
- `hits_dim_4d.tsv`
- `hits_taskdomains_dim.tsv`

Example OCR scans:

```bash
while IFS= read -r f; do
  rg -n -i -H "\\b1d\\b|one[- ]dimensional|token sequence|time series|audio waveform|language modeling|autoregressive text" "$f"
done < RUNNERS/keyword-sort/transformer_ocr_files.txt > RUNNERS/keyword-sort/hits_dim_1d.tsv

while IFS= read -r f; do
  rg -n -i -H "\\b2d\\b|two[- ]dimensional|image|pixel|patch|grid|table|board|maze|sudoku|\\(x,\\s*y\\)" "$f"
done < RUNNERS/keyword-sort/transformer_ocr_files.txt > RUNNERS/keyword-sort/hits_dim_2d.tsv

while IFS= read -r f; do
  rg -n -i -H "\\b3d\\b|three[- ]dimensional|point cloud|voxel|volume|mesh|video|\\(x,\\s*y,\\s*z\\)|\\(x,\\s*y,\\s*t\\)" "$f"
done < RUNNERS/keyword-sort/transformer_ocr_files.txt > RUNNERS/keyword-sort/hits_dim_3d.tsv

while IFS= read -r f; do
  rg -n -i -H "\\b4d\\b|four[- ]dimensional|\\(x,\\s*y,\\s*z,\\s*t\\)|spatiotemporal volume|space[- ]time volume" "$f"
done < RUNNERS/keyword-sort/transformer_ocr_files.txt > RUNNERS/keyword-sort/hits_dim_4d.tsv
```

Task file confirmation scan:

```bash
while IFS= read -r f; do
  rg -n -i -H "\\b1D\\b|\\b2D\\b|\\b3D\\b|\\b4D\\b" "$f"
done < RUNNERS/keyword-sort/transformer_taskdomain_csv_files.txt > RUNNERS/keyword-sort/hits_taskdomains_dim.tsv
```

Repeat for `transformer_taskdomain_md_files.txt` and append to the same file if desired.

### Step 4: Per-Paper Dimension Decision
For each paper, create:
- `ocr_dims`: dimensions evidenced from OCR
- `task_dims`: dimensions evidenced from task files
- `final_dims`: union with OCR-preferred conflict resolution

Final label:
- exactly one dimension in `final_dims` -> `<dim>_only`
- more than one dimension -> `multi-D`
- none -> `uncertain`

Confidence:
- `high`: OCR and task files agree
- `medium`: OCR-only or small mismatch
- `low`: task-files-only or ambiguous OCR language

### Step 5: Produce Outputs
Create:
- `RUNNERS/keyword-sort/transformer_task_dimensions_results.csv`
- `RUNNERS/keyword-sort/transformer_task_dimensions_summary.md`

Suggested CSV schema:

```csv
paper_dir,step1_label,final_label,final_dims,ocr_dims,task_dims,confidence,ocr_evidence,task_evidence
```

Where:
- `final_label` in `{1D_only,2D_only,3D_only,4D_only,multi-D,uncertain}`
- `final_dims` is semicolon-separated set like `2D;3D`

## Quality Checks
- Every included paper has at least one OCR evidence line in `ocr_evidence`.
- No paper marked `1D_only` solely because text says `token sequence` while task files show 2D/3D grids.
- Multi-task papers with mixed task dimensions are marked `multi-D`.
- Spot-check 10-20% of papers manually before finalizing.

## Optional Add-On
Add columns for positional encoding heterogeneity among transformer papers:
- `rope_used`
- `relative_pe_used`
- `absolute_pe_used`
- `learned_pe_used`
- `pe_notes`

This directly supports your broader claim about non-consensus PE choices across >1D tasks.

