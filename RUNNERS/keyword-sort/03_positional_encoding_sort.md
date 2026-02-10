# Stage 3: Positional Encoding Sort

## Goal
For Stage-2 papers labeled `2D_only`, `3D_only`, `4D_only`, or `multi-D`, classify positional encoding family.

## Method
- OCR pattern matching for: `rope`, `learned_absolute`, `sinusoidal_absolute`, `relative_position`, `alibi`, `other_variant`, `none_or_implicit`, `mixed`.
- Context and baseline filtering reduce false positives from related-work/reference mentions.
- Fallback pass captures weaker but useful PE phrasing.
- Conservative title priors are used for known backbone families when explicit PE text is sparse.

## Output
- `positional_encoding_results.csv`

## Final Counts
- total: `115`
- `learned_absolute`: `16`
- `rope`: `6`
- `relative_position`: `6`
- `sinusoidal_absolute`: `3`
- `other_variant`: `11`
- `mixed`: `1`
- `alibi`: `1`
- `none_or_implicit`: `1`
- `unclear`: `70`

## Notes
- `unclear` mostly indicates inherited/unstated PE, or papers where PE is not the core contribution.
