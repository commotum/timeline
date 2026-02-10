# Positional Encoding Summary

Final post-refinement counts (from positional_encoding_results.csv):
2D+/multi-D transformer papers processed: 115

PE labels:
- alibi: 1
- learned_absolute: 16
- mixed: 1
- none_or_implicit: 1
- other_variant: 11
- relative_position: 6
- rope: 6
- sinusoidal_absolute: 3
- unclear: 70

Top dimension x PE combinations:
- multi-D x unclear: 52
- 2D_only x unclear: 17
- multi-D x learned_absolute: 11
- multi-D x other_variant: 10
- multi-D x relative_position: 5
- multi-D x rope: 5
- 2D_only x learned_absolute: 5
- 2D_only x sinusoidal_absolute: 2
- 2D_only x rope: 1
- 3D_only x other_variant: 1
- 2D_only x relative_position: 1
- multi-D x sinusoidal_absolute: 1
- multi-D x alibi: 1
- multi-D x none_or_implicit: 1
- 3D_only x unclear: 1

Canonical file:
- positional_encoding_results.csv
