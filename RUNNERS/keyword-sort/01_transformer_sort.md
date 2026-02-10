# Stage 1: Transformer Sort

## Goal
Classify each paper in `BIBLIOTHEQUE/03_COMP-REAS` and `BIBLIOTHEQUE/05_ML-FNDTNS` as transformer-family, non-transformer, hybrid, or uncertain using OCR `.md` text.

## Method
- Strong transformer cues (`A`): transformer/ViT/GPT/BERT/etc.
- Attention cues (`B`): self-attention and variants.
- Support cues (`C`): token/context/positional terms.
- Classic non-transformer cues (`D`): RNN/CNN/Q-learning/actor-critic/etc.
- Context weighting avoids counting references/related-work noise.
- Added guard for Atari `Q*Bert` false positives so RL game tables do not trigger BERT matches.

## Output
- `transformer_screen_results.csv`

## Final Counts
- total: `423`
- `transformer_yes`: `179`
- `hybrid_transformer_yes`: `50`
- `transformer_no`: `122`
- `uncertain`: `72`

## Notes
- `transformer_yes + hybrid_transformer_yes = 229` papers move to Stage 2.
