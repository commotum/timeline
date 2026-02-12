# Dimensionality Binary Sort Runner

This runner creates exactly one file per paper folder:
- `DIMENSION-1D.md` or
- `DIMENSION-2DPLUS.md`

Target scope (default):
- `BIBLIOTHEQUE/03_COMP-REAS`
- `BIBLIOTHEQUE/05_ML-FNDTNS`

## Files examined per paper folder

Primary (expensive fallback):
1. `<Paper Folder Name>.md` (full OCR markdown of the paper)

Hint-only (non-authoritative; used for navigation speed):
2. `TASK-DOMAINS.md`
3. `TASK-DOMAINS.csv`
4. `TASK_MODEL_RATIO.md`
5. `TRANSFORMER-YES.md`
6. `TRANSFORMER-NO.md`

Token-saving policy:
- The prompt uses hint-first triage.
- It only reads the primary OCR markdown when hints are not enough.

## Decision flow

1. Pass 0: hint-first triage.
2. Pass 1: source triage only if needed.
3. Pass 2: source deep dive only if still unresolved.

No uncertain output is allowed. The runner must choose 1D or 2D+.

## Usage

From repo root:

```bash
python3 RUNNERS/dimensionality-binary-sort/script-dimensionality-binary-sort.py
```

Useful options:

```bash
# Re-run everything and overwrite prior files
python3 RUNNERS/dimensionality-binary-sort/script-dimensionality-binary-sort.py --overwrite

# Dry-run prompt generation only
python3 RUNNERS/dimensionality-binary-sort/script-dimensionality-binary-sort.py --dry-run

# Run a single paper folder
python3 RUNNERS/dimensionality-binary-sort/script-dimensionality-binary-sort.py \
  --folders "BIBLIOTHEQUE/03_COMP-REAS/<Paper Folder Name>"
```

## Operational files

- Prompt template: `prompt-dimensionality-binary-sort.md`
- Script: `script-dimensionality-binary-sort.py`
- State file (created on run): `.codex_dimensionality_binary_state.json`
- Log file (created on run): `codex_dimensionality_binary.log`
