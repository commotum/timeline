# Keyword Screen Plan: Transformer/Attention Usage

## Goal
Identify which papers in:
- `BIBLIOTHEQUE/03_COMP-REAS`
- `BIBLIOTHEQUE/05_ML-FNDTNS`

should be counted as **Transformer-family** (including hybrid models) based on keyword evidence in `.md` and `.csv` files only.

## Scope and Constraints
- Allowed inputs: `.md` and `.csv` files only.
- Primary evidence source per paper: OCR markdown at `<paper_dir>/<paper_dir_name>.md`.
- Runner outputs (`CLASS_*.md`, `TASK-DOMAINS.md`, `TASK_MODEL_RATIO.md`) can be used as secondary hints, not primary proof.
- Ignore PDFs and images completely.

## Counting Policy
Count a paper as Transformer-family if:
1. It explicitly uses a Transformer architecture, or
2. It explicitly uses self-attention/windowed/sparse/axial/cross attention in a roughly Transformer-like model.

Count a paper as **Hybrid Transformer** if:
- It combines Transformer/attention blocks with RNN/CNN/RL/other modules, and
- Transformer/attention is part of the core model (not only a citation or baseline mention).

Do **not** count as Transformer-family if:
- Only classical families are used (RNN/LSTM/GRU/CNN/pure RL/etc.), and
- No credible Transformer/attention evidence appears in the method/model sections.

## Labels
Use one label per paper:
- `transformer_yes`
- `hybrid_transformer_yes`
- `transformer_no`
- `uncertain`

## Keyword Library

### A) Strong Transformer Cues (high precision)
- `transformer`
- `vision transformer`
- `encoder-only transformer`
- `decoder-only transformer`
- `encoder-decoder transformer`
- `ViT`
- `GPT`
- `BERT`
- `RoFormer`
- `Swin Transformer`

### B) Attention Mechanism Cues (count if architecture-context is clear)
- `self-attention`
- `multi-head attention`
- `scaled dot product attention`
- `cross-attention`
- `windowed attention`
- `local attention`
- `sparse attention`
- `axial attention`
- `hierarchical attention`
- `causal attention`
- `flash attention` / `flashattention`

### C) Transformer-Adjacent Supporting Cues
- `token`
- `tokenization`
- `context length`
- `positional encoding`
- `positional embedding`
- `rotary` / `RoPE`
- `relative position`
- `absolute position embedding` / `APE`
- `QKV`

### D) Non-Transformer Family Cues
- `recurrent neural network` / `RNN`
- `LSTM`
- `GRU`
- `convolutional neural network` / `CNN`
- `policy gradient`
- `Q-learning`
- `actor-critic`
- `PPO`
- `DQN`
- `SARSA`

## Decision Rules
For each paper, compute keyword hit counts from OCR `.md`:
- `A_hits`: Strong Transformer cues
- `B_hits`: Attention mechanism cues
- `C_hits`: Supporting cues
- `D_hits`: Non-Transformer cues

Classification:
1. `transformer_yes` if `A_hits >= 1`, or (`B_hits >= 2` and `C_hits >= 2` with architecture context).
2. `hybrid_transformer_yes` if rule (1) is true and `D_hits >= 1` in core model description.
3. `transformer_no` if `A_hits == 0` and `B_hits == 0` and `D_hits >= 2`.
4. `uncertain` otherwise.

## Guardrails Against False Positives
- Ignore hits found only in references/bibliography.
- Ignore hits only in related work unless method/architecture sections also contain evidence.
- Require at least one direct evidence quote (line-cited) from model/method/architecture/training sections.

## Low-Token Workflow (Two Pass)

### Pass 1: Fast scan over OCR markdown files
Build OCR file list:

```bash
find BIBLIOTHEQUE/03_COMP-REAS BIBLIOTHEQUE/05_ML-FNDTNS -mindepth 1 -maxdepth 1 -type d -print0 \
| while IFS= read -r -d '' dir; do
  stem="$(basename "$dir")"
  ocr="$dir/$stem.md"
  if [[ -f "$ocr" ]]; then
    printf '%s\n' "$ocr"
  fi
done > RUNNERS/keyword-sort/paper_ocr_list.txt
```

Run keyword scans (line-numbered evidence):

```bash
while IFS= read -r f; do
  rg -n -i -H "transformer|vision transformer|encoder-only transformer|decoder-only transformer|vit|gpt|bert|roformer|swin transformer" "$f"
done < RUNNERS/keyword-sort/paper_ocr_list.txt > RUNNERS/keyword-sort/hits_A_strong.tsv

while IFS= read -r f; do
  rg -n -i -H "self-attention|multi-head attention|scaled dot product attention|cross-attention|windowed attention|local attention|sparse attention|axial attention|hierarchical attention|causal attention|flash ?attention" "$f"
done < RUNNERS/keyword-sort/paper_ocr_list.txt > RUNNERS/keyword-sort/hits_B_attention.tsv

while IFS= read -r f; do
  rg -n -i -H "token|tokenization|context length|positional encoding|positional embedding|rotary|rope|relative position|absolute position embedding|\\bAPE\\b|\\bQKV\\b" "$f"
done < RUNNERS/keyword-sort/paper_ocr_list.txt > RUNNERS/keyword-sort/hits_C_support.tsv

while IFS= read -r f; do
  rg -n -i -H "recurrent neural network|\\bRNN\\b|\\bLSTM\\b|\\bGRU\\b|convolutional neural network|\\bCNN\\b|policy gradient|q-learning|actor-critic|\\bPPO\\b|\\bDQN\\b|\\bSARSA\\b" "$f"
done < RUNNERS/keyword-sort/paper_ocr_list.txt > RUNNERS/keyword-sort/hits_D_classic.tsv
```

### Pass 2: Targeted verification only for uncertain/edge papers
- For papers with conflicting signals, open only hit neighborhoods (`line-3` to `line+3`) to confirm context.
- Promote to `transformer_yes` or `hybrid_transformer_yes` only if method-level evidence is clear.

## Output Artifacts
Create:
- `RUNNERS/keyword-sort/transformer_screen_results.csv`
- `RUNNERS/keyword-sort/transformer_screen_summary.md`

Suggested CSV schema:

```csv
paper_dir,class_code,A_hits,B_hits,C_hits,D_hits,label,confidence,evidence_lines
```

Confidence guide:
- `high`: explicit Transformer and/or explicit self-attention in model description.
- `medium`: multiple attention cues plus supporting cues, but wording is less direct.
- `low`: weak/ambiguous evidence; likely `uncertain`.

## Optional Extension (for positional-encoding analysis)
Track PE keywords per paper in extra columns:
- `rope`
- `relative_pe`
- `absolute_pe`
- `learned_pe`
- `axial_pe`

This gives a quick empirical map of PE heterogeneity after Transformer-family papers are identified.

