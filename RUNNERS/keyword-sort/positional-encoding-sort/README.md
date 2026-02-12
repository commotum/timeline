# Positional Encoding Mechanism Sort Runner

This runner creates exactly one PE label file per paper folder:
- `PE-ROPE.md`
- `PE-LEARNED-ABSOLUTE.md`
- `PE-SINUSOIDAL-ABSOLUTE.md`
- `PE-RELATIVE-POSITION.md`
- `PE-ALIBI.md`
- `PE-MIXED.md`
- `PE-NONE-OR-IMPLICIT.md`
- `PE-INHERITED-DEFAULT.md`
- `PE-UNCLEAR.md`

Target scope (default):
- `BIBLIOTHEQUE/03_COMP-REAS`
- `BIBLIOTHEQUE/05_ML-FNDTNS`

## Files examined per paper folder

Primary (expensive fallback):
1. `<Paper Folder Name>.md`

Hint-only (non-authoritative):
2. `TRANSFORMER-YES.md`
3. `TRANSFORMER-NO.md`
4. `DIMENSION-1D.md`
5. `DIMENSION-2DPLUS.md`
6. `TASK-DOMAINS.md`
7. `TASK-DOMAINS.csv`

Token-saving policy:
- The prompt uses hint-first triage.
- It only reads the primary OCR markdown when hints are not enough.

## Judgment policy

The prompt enforces 4 passes:
1. hint-first triage,
2. source keyword triage (only if needed),
3. source deep text dive (only if still needed),
4. inherited/default reasoning when explicit PE is missing.

The model is instructed to use knowledge of common backbone defaults carefully, and to mark inherited assumptions explicitly.

## Usage

```bash
python3 RUNNERS/positional-encoding-sort/script-positional-encoding-sort.py
```

Useful options:

```bash
# Re-run everything and overwrite prior PE labels
python3 RUNNERS/positional-encoding-sort/script-positional-encoding-sort.py --overwrite

# Dry-run prompt generation only
python3 RUNNERS/positional-encoding-sort/script-positional-encoding-sort.py --dry-run

# Run a single paper folder
python3 RUNNERS/positional-encoding-sort/script-positional-encoding-sort.py \
  --folders "BIBLIOTHEQUE/03_COMP-REAS/<Paper Folder Name>"
```

## Operational files

- Prompt template: `prompt-positional-encoding-sort.md`
- Script: `script-positional-encoding-sort.py`
- State file (created on run): `.codex_positional_encoding_sort_state.json`
- Log file (created on run): `codex_positional_encoding_sort.log`
