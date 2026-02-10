# Positional Encoding Summary

2D+/multi-D transformer papers processed: 117

PE labels:
- alibi: 1
- learned_absolute: 6
- mixed: 1
- none_or_implicit: 1
- other_variant: 12
- relative_position: 4
- rope: 6
- sinusoidal_absolute: 2
- unclear: 84

Top dimension x PE combinations:
- multi-D x unclear: 60
- 2D_only x unclear: 21
- multi-D x other_variant: 10
- multi-D x rope: 5
- multi-D x learned_absolute: 4
- 3D_only x unclear: 3
- multi-D x relative_position: 3
- 2D_only x learned_absolute: 2
- 2D_only x other_variant: 1
- 2D_only x relative_position: 1
- 2D_only x rope: 1
- 2D_only x sinusoidal_absolute: 1
- 3D_only x other_variant: 1
- multi-D x alibi: 1
- multi-D x mixed: 1
- multi-D x none_or_implicit: 1
- multi-D x sinusoidal_absolute: 1

Files generated:
- positional_encoding_results.csv
- pe_candidate_paper_dirs.txt
- pe_candidate_ocr_files.txt
- pe_hits_raw.tsv
- pe_hit_windows.tsv
