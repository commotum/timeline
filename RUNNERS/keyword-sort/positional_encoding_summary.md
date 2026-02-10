# Positional Encoding Summary

2D+/multi-D transformer papers processed: 212

PE labels:
- alibi: 1
- learned_absolute: 7
- mixed: 6
- none_or_implicit: 1
- other_variant: 13
- relative_position: 10
- rope: 9
- sinusoidal_absolute: 9
- unclear: 156

Top dimension x PE combinations:
- multi-D x unclear: 122
- 2D_only x unclear: 34
- multi-D x other_variant: 12
- multi-D x relative_position: 9
- multi-D x mixed: 6
- multi-D x rope: 6
- multi-D x sinusoidal_absolute: 6
- multi-D x learned_absolute: 4
- 2D_only x learned_absolute: 3
- 2D_only x rope: 3
- 2D_only x sinusoidal_absolute: 3
- 2D_only x other_variant: 1
- 2D_only x relative_position: 1
- multi-D x alibi: 1
- multi-D x none_or_implicit: 1

Files generated:
- positional_encoding_results.csv
- pe_candidate_paper_dirs.txt
- pe_candidate_ocr_files.txt
- pe_hits_raw.tsv
- pe_hit_windows.tsv
